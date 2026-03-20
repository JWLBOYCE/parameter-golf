from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

from data.download_hf_docs_and_tokenize import APPEND_EOS, NUM_VAL_DOCS, PureByteTokenizer, maybe_load_docs_sidecar_meta


def count_datafile_tokens(path: Path) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError(f"short header for {path}")
    return int(header[2])


def total_validation_tokens(pattern: str) -> int:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no validation shards found for pattern {pattern}")
    return sum(count_datafile_tokens(path) for path in files)


def validate_offsets_against_validation_shards(offsets: np.ndarray, val_pattern: str) -> int:
    total_tokens = total_validation_tokens(val_pattern)
    if int(offsets[-1]) != total_tokens:
        raise ValueError(
            f"validation doc offsets mismatch shard token count: offsets={int(offsets[-1])} shard_tokens={total_tokens}"
        )
    return total_tokens


def build_encoder(tokenizer_path: Path):
    if tokenizer_path.suffix == ".model":
        tok = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

        def encode_batch(texts: list[str]) -> list[list[int]]:
            return tok.encode(texts, out_type=int)

        return encode_batch, int(tok.bos_id())
    if tokenizer_path.suffix == ".json":
        payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        if payload.get("tokenizer_type") != "pure_byte":
            raise ValueError(f"unsupported tokenizer json {tokenizer_path}")
        tok = PureByteTokenizer(**payload["config"])
        return tok.encode_batch, int(tok.bos_id)
    raise ValueError(f"unsupported tokenizer path {tokenizer_path}; expected .model or pure-byte .json")


def iter_docs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def batched(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_doc_offsets(
    *,
    docs_jsonl: Path,
    tokenizer_path: Path,
    num_val_docs: int,
    batch_size: int,
) -> np.ndarray:
    encode_batch, bos_id = build_encoder(tokenizer_path)
    del bos_id
    offsets = [0]
    docs_seen = 0
    for texts in batched(iter_docs(docs_jsonl), batch_size):
        if docs_seen >= num_val_docs:
            break
        if docs_seen + len(texts) > num_val_docs:
            texts = texts[: num_val_docs - docs_seen]
        encoded_docs = encode_batch(texts)
        for encoded in encoded_docs:
            offsets.append(offsets[-1] + 1 + len(encoded) + int(APPEND_EOS))
            docs_seen += 1
    if docs_seen != num_val_docs:
        raise ValueError(f"expected {num_val_docs} validation docs, encoded {docs_seen}")
    return np.asarray(offsets, dtype=np.int64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fineweb_val_doc_offsets.npy from docs_selected.jsonl")
    parser.add_argument("--docs-jsonl", type=Path, default=Path("data/docs_selected.jsonl"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--val-files", default="data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--output", type=Path, default=Path("data/datasets/fineweb10B_sp1024/fineweb_val_doc_offsets.npy"))
    parser.add_argument("--num-val-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sidecar = maybe_load_docs_sidecar_meta(args.docs_jsonl)
    num_val_docs = (
        int(args.num_val_docs)
        if args.num_val_docs is not None
        else int(sidecar["docs_val"]) if sidecar is not None and sidecar.get("docs_val") is not None else NUM_VAL_DOCS
    )
    offsets = build_doc_offsets(
        docs_jsonl=args.docs_jsonl,
        tokenizer_path=args.tokenizer_path,
        num_val_docs=num_val_docs,
        batch_size=max(1, int(args.batch_size)),
    )
    total_tokens = validate_offsets_against_validation_shards(offsets, args.val_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, offsets)
    print(json.dumps({"output": str(args.output), "num_val_docs": num_val_docs, "total_tokens": total_tokens}, indent=2))


if __name__ == "__main__":
    main()
