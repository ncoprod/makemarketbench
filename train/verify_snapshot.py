"""Compare local model/tokenizer files with the immutable Hub revision's blobs."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from makemarket.provenance import receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        remote = json.load(response)
    record = {"provenance": receipt(vars(args), args.revision), "metadata_url": url, "files": {}}
    for entry in remote["siblings"]:
        name = entry["rfilename"]
        path = Path(args.local) / name
        if not path.is_file() or path.suffix not in (".json", ".safetensors", ".jinja", ".txt"):
            continue
        lfs = entry.get("lfs")
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        expected = lfs["sha256"] if lfs else entry["blobId"]
        record["files"][name] = {"bytes": path.stat().st_size, "hash": digest.hexdigest(),
                                  "hash_algorithm": "sha256" if lfs else "git-blob-sha1",
                                  "expected_hash": expected, "matches": digest.hexdigest() == expected}
    record["all_local_checked_files_match"] = bool(record["files"]) and all(
        f["matches"] for f in record["files"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print("All checked files match:", record["all_local_checked_files_match"])


if __name__ == "__main__":
    main()
