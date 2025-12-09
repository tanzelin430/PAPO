

import json

INPUT_JSONL = "test2.jsonl"
OUTPUT_JSONL = "test.jsonl"


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    count_in = 0
    count_out = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for rec in iter_jsonl(INPUT_JSONL):
            count_in += 1

            question = rec.get("question", "")
            candidates = rec.get("candidates", [])
            n_cand = rec.get("n_candidates", None)

            if not candidates:
                continue
            if n_cand is not None and n_cand != 1:
                continue

            cand = candidates[0]
            proof = cand.get("content", "")

            gt_obj = cand.get("GT_verification", {}) or {}
            gt_score = gt_obj.get("GT", None)
            gt_text = gt_obj.get("verification", None)

            meta_obj = cand.get("meta_verification", {}) or {}
            meta_gt = meta_obj.get("meta_GT", None)

            new_rec = {
                "question": question,
                "proof": proof,
                "GT_verify": {
                    "GT_score": gt_score,
                    "GT_verification": gt_text,
                },
                "GT_meta": meta_gt,
            }

            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"Number of original samples read: {count_in}")
    print(f"Number of converted samples written: {count_out}")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
