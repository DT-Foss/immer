"""s31_llama.py — S3.1-Sonden für Qwen3.8-27B via llama.cpp (CPU/GGUF).

Standalone für beast: nutzt llama-cpp-python + unsloth GGUF (Q4_K_M).
Gleiche Aufgaben-Batterie wie s31_sonden_27b.py (transformers-Version).

    python3 s31_llama.py --model /root/o1x_data/qwen38-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import json
import random
import time

SEED = 7

TASKS = {
    "add2": ("17+25=42\n38+14=52\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", str(a + b)))(r.randint(12, 88), r.randint(11, 87))),
    "add3": ("123+456=579\n234+567=801\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", str(a + b)))(r.randint(100, 499), r.randint(100, 499))),
    "add4": ("1234+5678=6912\n2345+6789=9134\n",
             lambda r: (lambda a, b: (f"{a}+{b}=", str(a + b)))(r.randint(1000, 4999), r.randint(1000, 4999))),
    "mul2x1": ("12*3=36\n23*4=92\n",
               lambda r: (lambda a, b: (f"{a}*{b}=", str(a * b)))(r.randint(12, 98), r.randint(2, 9))),
    "mul2x2": ("12*13=156\n23*14=322\n",
               lambda r: (lambda a, b: (f"{a}*{b}=", str(a * b)))(r.randint(12, 49), r.randint(12, 49))),
    "mul3x2": ("123*45=5535\n234*67=15678\n",
               lambda r: (lambda a, b: (f"{a}*{b}=", str(a * b)))(r.randint(101, 499), r.randint(12, 98))),
    "add3t": ("17+25+31=73\n38+14+22=74\n",
              lambda r: (lambda a, b, c: (f"{a}+{b}+{c}=", str(a + b + c)))(
                  r.randint(12, 88), r.randint(11, 87), r.randint(11, 87))),
    "sub3": ("456-178=278\n823-457=366\n",
             lambda r: (lambda a, b: (f"{a}-{b}=", str(a - b)))(
                 r.randint(300, 899), r.randint(100, 299))),
    "twostep": ("Example: a=3, b=a+2, c=b*2 -> c=10\n"
                "Example: a=5, b=a+1, c=b*3 -> c=18\n",
                lambda r: (lambda a: (f"a={a}, b=a+2, c=b*2 -> c=", str((a + 2) * 2)))(
                    r.randint(1, 9))),
    "wordlen": ("Example: length of 'hello' is 5\nExample: length of 'world' is 5\n",
                lambda r: (lambda w: (f"length of '{w}' is ", str(len(w))))(
                    "".join(random.Random(r.randrange(999)).choice("abcdefghij")
                            for _ in range(r.randint(3, 8))))),
    # --- S3.1b: haertere Batterie (der 27B war auf S3.1 gesaettigt) ---
    "add5": ("",
             lambda r: (lambda a, b: (f"{a}+{b}=", str(a + b)))(
                 r.randint(10000, 49999), r.randint(10000, 49999))),
    "mul3x3": ("",
               lambda r: (lambda a, b: (f"{a}*{b}=", str(a * b)))(
                   r.randint(101, 499), r.randint(101, 499))),
    "chain4": ("",
               lambda r: (lambda a: (f"a={a}, b=a+3, c=b*2, d=c-5, e=d*3 -> e=",
                                     str((((a + 3) * 2) - 5) * 3)))(
                   r.randint(1, 9))),
    "parens": ("Example: balance of '(())' is 1\nExample: balance of '())((' is 0\n",
               lambda r: (lambda s: (f"balance of '{s}' is ", "1" if s.count("(") == s.count(")") else "0"))(
                   "".join(random.Random(r.randrange(999)).choice("()")
                           for _ in range(r.randint(4, 10))))),
    "digitsum": ("",
                 lambda r: (lambda a: (f"digit sum of {a} is ", str(sum(int(c) for c in str(a)))))(
                     r.randint(1000, 99999))),
    # --- A5-Aufklaerung: prime/div3 in 3 Formaten (0-shot wie Sammlung /
    #     few-shot / 0-1-Ziffernraum). Range identisch zur A5-Sammlung. ---
    "prime0": ("",
               lambda r: (lambda a: (f"is {a} prime? ", "yes" if _is_prime(a) else "no"))(
                   r.randint(2, 199))),
    "prime_fs": ("is 11 prime? yes\nis 12 prime? no\nis 29 prime? yes\nis 91 prime? no\n",
                 lambda r: (lambda a: (f"is {a} prime? ", "yes" if _is_prime(a) else "no"))(
                     r.randint(2, 199))),
    "prime01": ("prime(11)=1\nprime(12)=0\nprime(29)=1\nprime(91)=0\n",
                lambda r: (lambda a: (f"prime({a})=", "1" if _is_prime(a) else "0"))(
                    r.randint(2, 199))),
    "div30": ("",
              lambda r: (lambda a: (f"is {a} divisible by 3? ", "yes" if a % 3 == 0 else "no"))(
                  r.randint(10, 999))),
    "div3_fs": ("is 12 divisible by 3? yes\nis 13 divisible by 3? no\n"
                "is 81 divisible by 3? yes\nis 92 divisible by 3? no\n",
                lambda r: (lambda a: (f"is {a} divisible by 3? ", "yes" if a % 3 == 0 else "no"))(
                    r.randint(10, 999))),
    "div301": ("div3(12)=1\ndiv3(13)=0\ndiv3(81)=1\ndiv3(92)=0\n",
               lambda r: (lambda a: (f"div3({a})=", "1" if a % 3 == 0 else "0"))(
                   r.randint(10, 999))),
    # mod3: echte Standard-Notation, Antwortraum {0,1,2} im Ziffernraum —
    # Hypothese: div301 (0,375) scheiterte an der Divisions-Lesart div3(12)->4
    "mod3": ("mod3(12)=0\nmod3(13)=1\nmod3(81)=0\nmod3(92)=2\n",
             lambda r: (lambda a: (f"mod3({a})=", str(a % 3)))(
                 r.randint(2, 999))),
    # addw: Wort-Zahlen-Addition — Antwortraum fuer den o1-state-Einbau
    # (dessen Vokabular ist wortbasiert, [a-zA-Z]+, KEINE Ziffern!)
    "addw": ("three plus two is five\nfour plus four is eight\n"
             "six plus three is nine\nseven plus five is twelve\n",
             lambda r: (lambda a, b: (f"{_W[a]} plus {_W[b]} is ", _W[a + b]))(
                 r.randint(2, 9), r.randint(2, 9))),
}

_W = ("zero one two three four five six seven eight nine ten eleven twelve "
      "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()


def _is_prime(n: int) -> bool:
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return n >= 2


def check(text: str, ans: str) -> bool:
    """Accepting-Parser (digitsum-Lektion): yes/no per Wort-Match,
    einstellige 0/1-Antworten per erster Ziffer, sonst alle Ziffern."""
    import re
    first = text.split("\n")[0].strip().lower()
    if ans in ("yes", "no"):
        m = re.search(r"\b(yes|no)\b", first)
        return bool(m) and m.group(1) == ans
    if ans in ("0", "1"):
        m = re.search(r"\d", first)
        return bool(m) and m.group(0) == ans
    if ans.isalpha():
        m = re.search(r"[a-z]+", first)
        return bool(m) and m.group(0) == ans
    digits = "".join(ch for ch in first if ch.isdigit())
    return digits == ans


def make(few, gen, n, seed):
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        q, ans = gen(rng)
        tasks.append((few + q, ans))
    return tasks


def main() -> None:
    from llama_cpp import Llama

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="sonden_27b_llama.json")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(SEED)
    print(f"lade {args.model} …", flush=True)
    llm = Llama(model_path=args.model, n_ctx=2048, n_threads=args.threads,
                n_batch=128, verbose=False)
    print(f"geladen ({time.time()-t0:.0f}s)", flush=True)

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    names = only or list(TASKS)
    res = {}
    for name in names:
        few, gen = TASKS[name]
        tasks = make(few, gen, args.n, SEED)
        ok = 0
        for prompt, ans in tasks:
            out = llm(prompt, max_tokens=8, temperature=0.0, echo=False)
            text = out["choices"][0]["text"]
            ok += int(check(text, ans))
        res[name] = round(ok / len(tasks), 3)
        print(f"  {name:10s}: acc {res[name]:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    res["meta"] = {"model": args.model, "n": args.n,
                   "runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"-> {args.out} | gesamt {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
