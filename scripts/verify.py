"""End-to-end verification against a running server. Usage:
    python scripts/verify.py [base_url]

Assumes: infra up (qdrant + mongo), collection created, server running.
Covers: ingest idempotency, raw-doc store idempotency, version supersession,
old-version guard, chat intents, 400/404 semantics.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def request(method: str, path: str, payload=None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def clause(mongo_id: str, year: str, em_r_lx: float, extra: str = "") -> dict:
    return {
        "_id": mongo_id,
        "standard_metadata": {
            "standard_code": "EN 12464-1", "version_year": year, "is_latest": True,
        },
        "hierarchy": {
            "category_table_number": "6.1", "category_title": "Corridors",
            "ref_number": "6.1.1", "page": 26,
        },
        "activity": "Corridors and circulation areas",
        "parameters": {"em_r_lx": em_r_lx, "uo": 0.4, "ra": 80, "ugr_rugl": 25,
                       "em_u_lx": None, "ez_lx": None, "em_wall_lx": None,
                       "em_ceiling_lx": None},
        "specific_requirements": "Uniform illumination along the passageway.",
        "searchable_text": (
            f"EN 12464-1 corridor lighting: maintained illuminance {em_r_lx} lx, "
            f"uniformity 0.40, Ra 80, UGRL 25." + extra
        ),
    }


def statuses(result: dict) -> dict[str, int]:
    counts = {}
    for item in result["results"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def cleanup():
    import re
    from app.config import Config
    from pymongo import MongoClient
    from app.utils.qdrant import create_qdrant_client

    mongo = MongoClient(Config.MONGO_URI)[Config.MONGO_DB]
    mongo["standards_clauses"].delete_many({"_id": {"$regex": "^(verify_|raw_)"}})

    client = create_qdrant_client()

    to_delete, offset = [], None
    while True:
        page, offset = client.scroll(Config.QDRANT_COLLECTION, limit=100, offset=offset, with_payload=True)
        for p in page:
            meta = p.payload.get("metadata") or p.payload
            if re.match(r"^(verify_|raw_)", meta.get("mongo_id", "")):
                to_delete.append(p.id)
        if not offset:
            break
    if to_delete:
        client.delete(Config.QDRANT_COLLECTION, points_selector=to_delete)


def main():
    cleanup()

    print("== health ==")
    status, body = request("GET", "/api/v1/health", None)
    check("health 200", status == 200, str(body))

    print("== ingest idempotency ==")
    doc1 = clause("verify_en12464_1_v2019_6_1_1", "2019", 100)
    status, body = request("POST", "/api/v1/ingest", {"documents": [doc1]})
    check("ingest #1 inserted", status == 200 and statuses(body) == {"inserted": 1}, body)
    status, body = request("POST", "/api/v1/ingest", {"documents": [doc1]})
    check("ingest #2 skipped_unchanged", statuses(body) == {"skipped_unchanged": 1}, body)
    doc1["searchable_text"] += " updated."
    status, body = request("POST", "/api/v1/ingest", {"documents": [doc1]})
    check("ingest #3 updated", statuses(body) == {"updated": 1}, body)
    check("no error on ingest", all(r.get("error") is None for r in body["results"]))

    print("== batch partial failure ==")
    bad = dict(doc1)
    bad["_id"] = "verify_bad_doc"
    bad.pop("searchable_text")
    status, body = request("POST", "/api/v1/ingest", {"documents": [doc1, bad]})
    counts = statuses(body)
    ok_count = counts.get("inserted", 0) + counts.get("skipped_unchanged", 0) + counts.get("updated", 0)
    check("batch: one failed, batch survives",
          counts.get("failed") == 1 and ok_count == 1, body)

    print("== raw /documents store ==")
    raw = clause("raw_note_1", "2019", 100)
    status, body = request("POST", "/api/v1/documents", [raw])
    check("store #1 inserted", status == 200 and statuses(body) == {"inserted": 1}, body)
    status, body = request("POST", "/api/v1/documents", [raw])
    check("store #2 skipped_unchanged", statuses(body) == {"skipped_unchanged": 1}, body)
    changed = dict(raw)
    changed["parameters"] = dict(raw["parameters"])
    changed["parameters"]["em_r_lx"] = 110
    status, body = request("POST", "/api/v1/documents", [changed])
    check("store #3 updated", statuses(body) == {"updated": 1}, body)
    noid = {"no_id_here": 1}
    status, body = request("POST", "/api/v1/documents", [noid])
    check("store missing _id failed", statuses(body) == {"failed": 1}, body)

    print("== versioning: old-version guard ==")
    v2021 = clause("verify_en12464_1_v2021_6_1_1", "2021", 150)
    status, body = request("POST", "/api/v1/ingest", {"documents": [v2021]})
    check("v2021 inserted", statuses(body) == {"inserted": 1}, body)
    old = clause("verify_en12464_1_v2019_6_1_1", "2019", 100)
    status, body = request("POST", "/api/v1/ingest", {"documents": [old]})
    check("v2019 re-ingest accepted (no error)",
          body["results"][0]["status"] != "failed", body)

    from pymongo import MongoClient
    from app.config import Config
    db = MongoClient(Config.MONGO_URI)[Config.MONGO_DB]
    col = db["standards_clauses"]
    d19 = col.find_one({"_id": "verify_en12464_1_v2019_6_1_1"})
    d21 = col.find_one({"_id": "verify_en12464_1_v2021_6_1_1"})
    check("older v2019 stays is_latest=false",
          d19 is not None and d19["standard_metadata"]["is_latest"] is False,
          d19 and d19["standard_metadata"])
    check("newer v2021 is is_latest=true",
          d21 is not None and d21["standard_metadata"]["is_latest"] is True,
          d21 and d21["standard_metadata"])

    print("== chat semantics ==")
    status, body = request("POST", "/api/v1/chat/message", {"message": "   "})
    check("empty message -> 400", status == 400, str(status))

    status, body = request("POST", "/api/v1/chat/message", {"message": "hi"})
    check("greeting intent", status == 200 and body.get("intent") == "greeting", body)

    sid = body.get("session_id")
    status, body = request("GET", f"/api/v1/chat/{sid}", None)
    check("session history has 2 turns",
          status == 200 and len(body.get("messages", [])) == 2, body)

    status, body = request("GET", "/api/v1/chat/nonexistent-session", None)
    check("unknown session -> 404", status == 404, str(status))

    status, body = request("POST", "/api/v1/chat/message",
                           {"session_id": sid, "message": "compare 2019 and 2021 corridors"})
    check("comparison deferred fallback",
          status == 200 and body.get("intent") == "comparison", body)
    status, body = request("GET", f"/api/v1/chat/{sid}", None)
    check("history grew by 2 turns",
          status == 200 and len(body.get("messages", [])) == 4, body)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()