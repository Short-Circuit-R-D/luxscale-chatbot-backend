from pydantic import BaseModel
from typing import Optional, Literal


class IntentEntities(BaseModel):
    standard_code: Optional[str] = None
    version_year: Optional[str] = None
    ref_number: Optional[str] = None


class IntentPrediction(BaseModel):
    intent: Literal[
        "greeting", "standard_query", "historical_query",
        "comparison", "out_of_scope", "fallback",
    ]
    entities: IntentEntities = IntentEntities()