"""Check that every teacher target follows the exact inference prompt prefix."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--data", default="train/data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    result = {"provenance": receipt(vars(args), args.revision), "splits": {}}
    for split in ("train", "valid"):
        path = Path(args.data) / f"{split}.jsonl"
        lengths, mismatches = [], []
        for index, line in enumerate(path.read_text().splitlines()):
            messages = json.loads(line)["messages"]
            prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True,
                                                   enable_thinking=False, return_dict=False)
            target = tokenizer.apply_chat_template(messages, enable_thinking=False, return_dict=False)
            if target[:len(prompt)] != prompt or len(target) <= len(prompt):
                mismatches.append(index)
            lengths.append(len(target))
        result["splits"][split] = {"rows": len(lengths), "max_tokens": max(lengths),
                                   "over_1536_tokens": sum(n > 1536 for n in lengths),
                                   "prefix_mismatch_rows": mismatches, "sha256": file_sha256(path)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    assert all(not s["prefix_mismatch_rows"] for s in result["splits"].values())


if __name__ == "__main__":
    main()
