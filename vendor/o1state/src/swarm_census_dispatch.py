"""DeepSeek swarm dispatcher: quality census over structure->text pairs.

Runs on the x86 data host. Shards the pairs file into batches, sends each
batch to deepseek-v4-flash via the pi CLI (Chinese instructions, thinking max,
no tools), collects strict-JSON labels. Every Nth batch is sent twice for an
agreement measurement. API-bound; negligible CPU next to the sweeps.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT_HEAD = (
    "你是英语教学语料的质量评审员。下面是若干条 structure→text 训练对。"
    "structure 描述一个因果三元组，text 应当是一句语法正确、与该三元组一致的英语句子。\n"
    "对每一条独立评审，标签定义：\n"
    "clean = 语法正确且与 structure 一致；\n"
    "wobbly = 可以理解但有语法瑕疵（如主谓不一致、用词生硬）；\n"
    "broken = 句子不成立（碎片、无主语、与 structure 矛盾）。\n"
    "reason_code 取值：ok | grammar | fragment | mismatch。\n"
    "请只输出一个 JSON 数组（不要其他任何文字），每条格式：\n"
    '{"i": <条目编号>, "label": "clean"|"wobbly"|"broken", "reason_code": "ok"|"grammar"|"fragment"|"mismatch"}\n\n'
)


def build_prompt(batch):
    lines = [PROMPT_HEAD]
    for j, (_gidx, p) in enumerate(batch):
        lines.append(f'[{j}] structure: {p["structure"]}')
        lines.append(f'    text: {p["text"]}')
    return "\n".join(lines)


MINI_SYSTEM = "你是严谨的语料质量评审员。只输出被要求的 JSON，不输出任何其他文字。"


def call_swarm(prompt, api_key, timeout_s):
    env = dict(os.environ, DEEPSEEK_API_KEY=api_key)
    r = subprocess.run(
        ["pi", "--provider", "deepseek", "--model", "deepseek-v4-flash",
         "--thinking", "max", "--no-tools", "--mode", "text",
         "--system-prompt", MINI_SYSTEM],
        input=prompt, capture_output=True, text=True, timeout=timeout_s, env=env,
    )
    out = r.stdout
    a, b = out.find("["), out.rfind("]")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("no JSON array in output")
    return json.loads(out[a:b + 1])


def grade_shard(shard_id, batch, _unused, api_key, timeout_s):
    prompt = build_prompt(batch)
    last_err = None
    for attempt in (1, 2):
        try:
            verdicts = call_swarm(prompt, api_key, timeout_s)
            rows = []
            for v in verdicts:
                j = int(v["i"])
                if 0 <= j < len(batch) and v.get("label") in ("clean", "wobbly", "broken"):
                    rows.append({"pair_idx": batch[j][0], "label": v["label"],
                                 "reason_code": v.get("reason_code", "?")})
            if len(rows) < 0.9 * len(batch):
                raise ValueError(f"only {len(rows)}/{len(batch)} verdicts parsed")
            return {"shard": shard_id, "rows": rows, "attempt": attempt}
        except Exception as e:  # noqa: BLE001 - a grader call may fail arbitrarily
            last_err = str(e)
            time.sleep(2)
    return {"shard": shard_id, "rows": [], "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--overlap-every", type=int, default=10)
    ap.add_argument("--timeout-s", type=int, default=240)
    ap.add_argument("--key-file", default=os.path.expanduser("~/.deepseek_env"))
    args = ap.parse_args()

    with open(args.key_file) as f:
        api_key = f.read().strip().split("=")[-1]
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            p = json.loads(line)
            pairs.append({"structure": p["structure"], "text": p["text"]})

    # Resume: skip every pair index already graded in a prior run (any shard
    # layout — coverage is by pair, not by shard id). Rebatch the remainder.
    graded = set()
    import glob
    for fp in glob.glob(os.path.join(args.out_dir, "census_*_main.json")):
        try:
            r = json.load(open(fp))
        except (OSError, json.JSONDecodeError):
            continue
        for row in r.get("rows", []):
            graded.add(row["pair_idx"])
    todo = [(i, p) for i, p in enumerate(pairs) if i not in graded]
    if graded:
        print(f"[census] resume: {len(graded)} pairs already graded, {len(todo)} to go")

    # Shard ids continue after the highest existing one so filenames never clash.
    existing = [int(os.path.basename(fp).split("_")[1])
                for fp in glob.glob(os.path.join(args.out_dir, "census_*_*.json"))]
    s0 = (max(existing) + 1) if existing else 0

    shards = []
    for k, start in enumerate(range(0, len(todo), args.batch_size)):
        chunk = todo[start:start + args.batch_size]
        s = s0 + k
        # grade_shard derives pair_idx as base_idx + position; with a sparse
        # todo list every batch carries its own explicit index map instead.
        shards.append((s, chunk, None, False))
        if args.overlap_every and k % args.overlap_every == 0:
            shards.append((s, chunk, None, True))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    done = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(grade_shard, s, b, base, api_key, args.timeout_s): (s, rep)
                for s, b, base, rep in shards}
        for fut in as_completed(futs):
            s, rep = futs[fut]
            res = fut.result()
            tag = "rep" if rep else "main"
            path = os.path.join(args.out_dir, f"census_{s:05d}_{tag}.json")
            with open(path, "w") as f:
                json.dump(res, f)
            done += 1
            n_fail += int(bool(res.get("error")))
            if done % 20 == 0 or done == len(shards):
                rate = done / (time.time() - t0)
                print(f"[census] {done}/{len(shards)} shards ({n_fail} failed) "
                      f"{rate:.2f} shards/s eta_min={(len(shards)-done)/max(rate,1e-9)/60:.0f}",
                      flush=True)
    print(f"[census] DONE shards={done} failed={n_fail} wall_s={time.time()-t0:.0f}")


if __name__ == "__main__":
    main()
