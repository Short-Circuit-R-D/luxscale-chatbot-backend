from pydantic import BaseModel
from typing import Optional, Literal
from .intents import Intent


class IntentEntities(BaseModel):
    standard_code: Optional[str] = None
    version_year: Optional[str] = None
    ref_number: Optional[str] = None


class IntentPrediction(BaseModel):
    intent: Intent
    entities: IntentEntities = IntentEntities()