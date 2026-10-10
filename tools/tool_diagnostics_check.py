"""CPU-only check of patch 0114: a whole GLM tool-call block naming a tool the request did not offer is kept as text
(no call is made) and reported: a log line naming the generated tool (never its arguments), and with
TF_GLM_TOOL_DIAGNOSTICS=1 a ``tool_call_rejections`` list in the reply's ``tensorfold`` block.

  python3 -B tools/tool_diagnostics_check.py --source-root /path/to/site-packages --model /path/to/model

``--model`` needs only the checkpoint's tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json and
generation_config.json (a model directory serves). No Torch/CUDA: the real
``App`` and HTTP handler serve on 127.0.0.1 over a replay engine that writes a fixed reply, token by token or in
rounds, so the reply goes through the same streaming and end parsers as a live one. Prints PASS/FAIL lines and
ALL PASS; exits non-zero on any failure."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

ARG = "text(1 + 1);"                   # an argument value: must never reach the log
CALL = f"<tool_call>exec<arg_key>input</arg_key><arg_value>{ARG}</arg_value></tool_call>"
REPLY = "I will check this." + CALL      # the issue's reply, verbatim
JSON_CALL = '<tool_call>{"name": "nope", "arguments": {"input": "' + ARG + '"}}</tool_call>'
OFFERED = "functions__exec"


def tool(name):
    return {"type": "function", "function": {"name": name, "description": "run code", "parameters": {
        "type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}}


class Replay:
    """An engine that writes ``reply`` (then its end token) in rounds of ``rounds`` tokens."""

    concurrent, tp, vision = False, 1, None

    def __init__(self, tok, eos):
        self.tok, self.eos = tok, eos
        self.reply, self.rounds = "", 1

    def generate(self, ids, count, sampling, feed, **_):
        out = [*self.tok.encode(self.reply, add_special_tokens=False).ids, self.eos[0]][:count]
        for at in range(0, len(out), self.rounds):
            if feed(out[at:at + self.rounds]):
                break
        return {}


FAILS: list[str] = []


def check(ok, label):
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    if not ok:
        FAILS.append(label)


def ask(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read().decode()
    if not body.get("stream"):
        reply = json.loads(data)
        msg = reply["choices"][0]["message"]
        return (msg.get("content") or "", msg.get("tool_calls") or [], reply["choices"][0]["finish_reason"],
                reply.get("tensorfold") or {})
    content, calls, finish, block = "", {}, None, {}
    for line in data.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if "tensorfold" in event:
            block = event["tensorfold"] or {}
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            for c in delta.get("tool_calls") or []:
                seen = calls.setdefault(c["index"], {"name": "", "arguments": ""})
                seen["name"] += (c.get("function") or {}).get("name") or ""
                seen["arguments"] += (c.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    return content, [{"function": v} for _, v in sorted(calls.items())], finish, block


def run(port, engine, reply, offered, *, stream, rounds, mode, thinking=False):
    """One request; (content, calls, finish, tensorfold block, the server's log lines)."""

    engine.reply, engine.rounds = reply, rounds
    if mode is None:
        os.environ.pop("TF_GLM_TOOL_DIAGNOSTICS", None)
    else:
        os.environ["TF_GLM_TOOL_DIAGNOSTICS"] = mode
    body = {"model": "m", "messages": [{"role": "user", "content": "add one and one"}], "tools": [tool(offered)],
            "stream": stream, "max_tokens": 512, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        got = ask(port, body)
    return (*got, [line for line in log.getvalue().splitlines() if "tool call" in line])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from http.server import ThreadingHTTPServer

    from tensorfold.cuda.http import make_handler
    from tensorfold.cuda.server import App

    model = Path(args.model)
    eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"] \
        if (model / "generation_config.json").exists() else None
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    eos = eos if isinstance(eos, list) else [tok.token_to_id("<|user|>")]
    engine = Replay(tok, eos)
    app = App(engine, model, "m", max_tokens=512)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    want = [{"reason": "unknown_tool_name", "name": "exec"}]
    try:
        for stream, rounds in ((False, 1), (True, 1), (True, 7), (True, 1024)):
            how = f"stream={stream} rounds={rounds}"
            # 1. the issue's case, diagnostics on: no call, the markup kept as text, the rejection reported
            content, calls, finish, block, log = run(port, engine, REPLY, OFFERED, stream=stream, rounds=rounds,
                                                     mode="1")
            check(not calls and finish == "stop", f"{how}: an unoffered name makes no call ({finish}, {calls})")
            check(CALL in content, f"{how}: the block stays in the content")
            check(block.get("tool_call_rejections") == want,
                  f"{how}: tensorfold.tool_call_rejections names it ({block.get('tool_call_rejections')})")
            check(len(log) == 1 and "unknown_tool_name" in log[0] and "'exec'" in log[0],
                  f"{how}: one log line names the generated tool ({log})")
            check(all(ARG not in line and "1 + 1" not in line for line in log), f"{how}: the log has no argument value")
            # 2. default (unset): the log line, the reply's shape unchanged
            content2, calls2, finish2, block2, log2 = run(port, engine, REPLY, OFFERED, stream=stream,
                                                          rounds=rounds, mode=None)
            check("tool_call_rejections" not in block2 and content2 == content and finish2 == finish,
                  f"{how}: default leaves the reply as it was")
            check(len(log2) == 1 and "'exec'" in log2[0], f"{how}: default still logs ({log2})")
            # 3. off: nothing
            _, _, _, block3, log3 = run(port, engine, REPLY, OFFERED, stream=stream, rounds=rounds, mode="0")
            check("tool_call_rejections" not in block3 and not log3, f"{how}: 0 turns it off ({log3})")
            # 4. the offered name: one call, nothing reported
            content4, calls4, finish4, block4, log4 = run(port, engine, REPLY, "exec", stream=stream, rounds=rounds,
                                                          mode="1")
            check(len(calls4) == 1 and calls4[0]["function"]["name"] == "exec"
                  and json.loads(calls4[0]["function"]["arguments"]) == {"input": ARG} and finish4 == "tool_calls",
                  f"{how}: an offered name is one call ({calls4}, {finish4})")
            check("tool_call_rejections" not in block4 and not log4 and CALL not in content4,
                  f"{how}: a call made reports nothing ({block4.get('tool_call_rejections')}, {log4})")
            # 5. JSON form and a thinking reply: the same report
            _, calls5, _, block5, log5 = run(port, engine, "ok " + JSON_CALL, OFFERED, stream=stream, rounds=rounds,
                                             mode="1")
            check(not calls5 and block5.get("tool_call_rejections") == [{"reason": "unknown_tool_name",
                                                                         "name": "nope"}]
                  and len(log5) == 1 and ARG not in log5[0], f"{how}: a JSON-form call to an unoffered tool too")
            _, calls6, _, block6, log6 = run(port, engine, "let me run it.</think>" + REPLY, OFFERED, stream=stream,
                                             rounds=rounds, mode="1", thinking=True)
            check(not calls6 and block6.get("tool_call_rejections") == want and len(log6) == 1,
                  f"{how}: thinking on, the same report ({block6.get('tool_call_rejections')})")
            # 6b. a block that is not a call at all stays text and is not reported as an unknown tool
            _, calls9, _, block9, log9 = run(port, engine, "see <tool_call>not a call at all</tool_call>", OFFERED,
                                             stream=stream, rounds=rounds, mode="1")
            check(not calls9 and "tool_call_rejections" not in block9 and not log9,
                  f"{how}: a block that is no call is not reported ({block9.get('tool_call_rejections')}, {log9})")
            # 7. two blocks, one offered and one not: one call, one rejection
            two = "a" + CALL.replace(">exec<", ">functions__exec<", 1) + "b" + CALL
            _, calls7, _, block7, log7 = run(port, engine, two, OFFERED, stream=stream, rounds=rounds, mode="1")
            check(len(calls7) == 1 and block7.get("tool_call_rejections") == want and len(log7) == 1,
                  f"{how}: an offered and an unoffered call: one call, one rejection ({calls7}, {log7})")
            # 8. a tool-call block outside a tools request is plain text, never reported
            engine.reply, engine.rounds = REPLY, rounds
            os.environ["TF_GLM_TOOL_DIAGNOSTICS"] = "1"
            log8 = io.StringIO()
            with contextlib.redirect_stdout(log8):
                _, calls8, _, block8 = ask(port, {"model": "m", "messages": [{"role": "user", "content": "x"}],
                                                  "stream": stream, "max_tokens": 512, "temperature": 0,
                                                  "chat_template_kwargs": {"enable_thinking": False}})
            check(not calls8 and "tool_call_rejections" not in block8 and "tool call" not in log8.getvalue(),
                  f"{how}: no tools offered, nothing reported")
    finally:
        server.shutdown()
    if FAILS:
        print(f"FAILED {len(FAILS)} checks")
        sys.exit(1)
    print("ALL PASS")


main()
