#!/usr/bin/env python3
"""gsm8k.py -- GSM8K test against a running llama-server (OpenAI chat API), as fixed in results/PREREGISTRATION.md: thinking off,
greedy, at most 768 tokens; the final answer is the number after '####', else the last number in the reply.

    python3 -I scripts/gsm8k.py --url http://127.0.0.1:8080 --data data/gsm8k/test.jsonl --out <dir>/gsm8k.jsonl [--limit N]
    python3 -I scripts/gsm8k.py --compare <A/gsm8k.jsonl> <B/gsm8k.jsonl>     paired: accuracy difference, McNemar exact

Writes one JSON line per question (resumes an interrupted run). Stdlib only.
"""
import argparse, json, math, random, re, sys, time, urllib.request

PROMPT = "{q}\nSolve this step by step, then give the final answer on its own line as: #### <number>"
NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def extract(text):
    m = re.findall(r"####\s*(" + NUM + ")", text)
    if not m:
        m = re.findall(NUM, text)
    if not m:
        return None
    s = m[-1].replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def gold(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def ask(url, q, max_tokens):
    body = {"model": "q", "messages": [{"role": "user", "content": PROMPT.format(q=q)}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(fa, fb):
    A = {r["idx"]: r for r in map(json.loads, open(fa))}
    B = {r["idx"]: r for r in map(json.loads, open(fb))}
    idx = sorted(set(A) & set(B))
    a = [A[i]["correct"] for i in idx]; b = [B[i]["correct"] for i in idx]
    only_a = sum(x and not y for x, y in zip(a, b)); only_b = sum(y and not x for x, y in zip(a, b))
    rng = random.Random(0); diffs = []
    for _ in range(4000):
        s = [rng.randrange(len(idx)) for _ in idx]
        diffs.append(sum(b[k] - a[k] for k in s) / len(idx))
    diffs.sort()
    print(f"{len(idx)} questions: A {sum(a)/len(idx)*100:.2f} %, B {sum(b)/len(idx)*100:.2f} %, B - A {(sum(b)-sum(a))/len(idx)*100:+.2f} points "
          f"[{diffs[100]*100:+.2f}, {diffs[3899]*100:+.2f}]; only A right {only_a}, only B right {only_b}, McNemar p = {mcnemar(only_a, only_b):.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--data"); ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=768); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    try:
        done = {json.loads(l)["idx"] for l in open(a.out)}
    except FileNotFoundError:
        pass
    n_ok = n = 0; t0 = time.time(); toks = 0
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            if i in done:
                continue
            res = ask(a.url, r["question"], a.max_tokens)
            text = res["choices"][0]["message"].get("content") or ""
            pred, g = extract(text), gold(r["answer"])
            ok = pred is not None and abs(pred - g) < 1e-6
            tm = res.get("timings") or {}
            rec = {"idx": i, "correct": ok, "pred": pred, "gold": g, "tokens": tm.get("predicted_n"),
                   "finish": res["choices"][0].get("finish_reason"), "tps": tm.get("predicted_per_second"), "text": text}
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1; n_ok += ok; toks += tm.get("predicted_n") or 0
            if n % 100 == 0:
                print(f"  {n} done, {n_ok/n*100:.1f} % right, {toks/(time.time()-t0):.0f} tok/s overall", flush=True)
    allr = [json.loads(l) for l in open(a.out)]
    print(f"gsm8k: {len(allr)} questions, {sum(r['correct'] for r in allr)/len(allr)*100:.2f} % right, "
          f"{sum(r['finish'] == 'length' for r in allr)} cut at {a.max_tokens} tokens", flush=True)


if __name__ == "__main__":
    sys.exit(main())
