from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


@dataclass(frozen=True)
class ClaraExample:
    data_type: str
    question: str
    docs: list[str]
    answer: str

    def to_json(self) -> dict[str, Any]:
        return {
            "data_type": self.data_type,
            "question": self.question,
            "docs": self.docs,
            "answer": self.answer,
        }


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _as_list_of_str(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [_as_str(v) for v in x if _as_str(v).strip()]
    if isinstance(x, tuple):
        return [_as_str(v) for v in x if _as_str(v).strip()]
    s = _as_str(x).strip()
    return [s] if s else []


def normalize_clara_record(record: dict[str, Any]) -> ClaraExample:
    """Normalize a row from `apple/CLaRa_multi_stage` into a stable schema.

    The upstream dataset has multiple stages/types; fields may vary slightly.
    We keep this tolerant so streaming + small debug subsets work reliably.
    """

    data_type = _as_str(record.get("data_type") or record.get("stage") or "unknown").strip()

    question = _as_str(record.get("question") or record.get("query") or record.get("instruction"))
    if isinstance(record.get("question"), list):
        # Some stage-1 formats store question as a list.
        q_list = _as_list_of_str(record.get("question"))
        question = q_list[0] if q_list else ""

    docs = _as_list_of_str(record.get("docs") or record.get("documents") or record.get("context"))

    # Answer fields differ by stage.
    answer = _as_str(
        record.get("gold_answer")
        or record.get("answer")
        or record.get("answers")
        or record.get("output")
        or ""
    )
    if isinstance(record.get("answers"), list):
        a_list = _as_list_of_str(record.get("answers"))
        answer = a_list[0] if a_list else ""

    return ClaraExample(
        data_type=data_type or "unknown",
        question=question.strip(),
        docs=[d.strip() for d in docs if d.strip()],
        answer=answer.strip(),
    )


def iter_clara_examples(
    *,
    dataset_name: str,
    split: str,
    streaming: bool,
    limit: int,
) -> Iterable[ClaraExample]:
    """Yield normalized examples from the Hugging Face dataset.

    Uses streaming mode to avoid downloading full shards.
    """

    if limit <= 0:
        return

    def _yield_from_hf_datasets() -> Iterable[ClaraExample]:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)

        count = 0
        for record in ds:
            if not isinstance(record, dict):
                continue
            ex = normalize_clara_record(record)
            if not ex.question and not ex.docs and not ex.answer:
                continue
            yield ex
            count += 1
            if count >= limit:
                break

    try:
        yield from _yield_from_hf_datasets()
        return
    except TypeError:
        # `apple/CLaRa_multi_stage` currently trips a Parquet casting error in some
        # versions of `datasets` when streaming (int64 -> null). Fall back to reading
        # just one parquet shard directly, which still avoids a full download.
        yield from _yield_from_single_parquet(dataset_name=dataset_name, split=split, limit=limit)


def export_debug_jsonl(
    *,
    out_path: str,
    dataset_name: str,
    split: str,
    streaming: bool,
    limit: int,
) -> int:
    """Export first N examples to JSONL for fast local iteration."""

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in iter_clara_examples(
            dataset_name=dataset_name,
            split=split,
            streaming=streaming,
            limit=limit,
        ):
            f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
            written += 1

    return written


def _pick_parquet_file(files: list[str], split: str) -> str:
    split_l = split.lower()
    parquet_files = [f for f in files if f.endswith(".parquet")]
    # Prefer split-specific paths.
    preferred = [
        f
        for f in parquet_files
        if (f"/{split_l}/" in f.lower())
        or (f"{split_l}-" in f.lower())
        or (f"_{split_l}_" in f.lower())
    ]
    if preferred:
        return sorted(preferred)[0]
    if parquet_files:
        return sorted(parquet_files)[0]
    raise RuntimeError("No parquet files found in dataset repo.")


def _yield_from_single_parquet(
    *, dataset_name: str, split: str, limit: int
) -> Iterable[ClaraExample]:
    files = list_repo_files(dataset_name, repo_type="dataset")
    parquet_path = _pick_parquet_file(files, split)
    local_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=parquet_path)

    pf = pq.ParquetFile(local_path)
    read = 0
    for batch in pf.iter_batches(batch_size=min(128, max(1, limit))):
        tbl = batch.to_pydict()
        # Convert columns dict-of-lists into row dicts.
        keys = list(tbl.keys())
        n = len(tbl[keys[0]]) if keys else 0
        for i in range(n):
            record = {k: tbl[k][i] for k in keys}
            ex = normalize_clara_record(record)
            if not ex.question and not ex.docs and not ex.answer:
                continue
            yield ex
            read += 1
            if read >= limit:
                return
