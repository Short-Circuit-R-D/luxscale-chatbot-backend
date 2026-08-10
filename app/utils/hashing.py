import hashlib
import json


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_json_hash(doc: dict) -> str:
    canonical = json.dumps(
        doc, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()