"""Seed sample EN 12464-1 clauses through the API. Usage:
    python scripts/seed_sample_data.py [base_url]

Expects the server on http://localhost:5000 (default). Posts three clauses
through POST /api/v1/ingest, including two editions of the same clause
(2019, then 2021) to exercise the supersession logic.
"""
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"


def corridor_clause(mongo_id: str, year: str, em_r_lx: float, uo: float,
           activity: str, ref: str, table: str, page: int,
           is_latest: bool, extra: str = "") -> dict:
    return {
        "_id": mongo_id,
        "qdrant_point_id": 0,
        "standard_metadata": {
            "standard_code": "EN 12464-1",
            "version_year": year,
            "is_latest": is_latest,
        },
        "hierarchy": {
            "category_table_number": table,
            "category_title": activity,
            "ref_number": ref,
            "page": page,
        },
        "activity": activity,
        "parameters": {"em_r_lx": em_r_lx, "uo": uo, "ra": 80,
                       "ugr_rugl": 25, "em_u_lx": None, "ez_lx": None,
                       "em_wall_lx": None, "em_ceiling_lx": None},
        "specific_requirements": extra or activity,
        "searchable_text": (
            f"EN 12464-1 lighting for {activity}: maintained illuminance "
            f"{em_r_lx} lx, minimum uniformity U0 {uo}, minimum colour "
            f"rendering index Ra 80, glare limit UGRL 25. {extra or activity}."
        ),
    }


DOCUMENTS = [
    corridor_clause("en12464_1_v2019_6_1_1", "2019", 100, 0.4,
                    "Corridors and circulation areas", "6.1.1", "6.1", 26, True,
                    "Uniform illumination along the passageway."),
    corridor_clause("en12464_1_v2021_6_1_1", "2021", 150, 0.4,
                    "Corridors and circulation areas", "6.1.1", "6.1", 28, True,
                    "Uniform illumination along the passageway."),
    corridor_clause("en12464_1_v2021_6_1_3", "2021", 200, 0.4,
                    "Entrance halls and reception areas", "6.1.3", "6.1", 28, True),
    corridor_clause("en12464_1_v2021_4_1_1", "2021", 300, 0.4,
                    "Offices - writing, typing, reading", "4.1.1", "4.1", 18, True,
                    ""),
]


def post(endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{endpoint}", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    result = post("/api/v1/ingest", {"documents": DOCUMENTS})
    for item in result["results"]:
        print(f'{item["status"]:<18} {item["mongo_id"]}')
    failed = [r for r in result["results"] if r["status"] == "failed"]
    if failed:
        print("FAILED:", failed, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()