from typing import Optional

from pydantic import BaseModel

from .intents import Intent


class IntentEntities(BaseModel):
    standard_code: Optional[str] = None
    version_year: Optional[str] = None
    ref_number: Optional[str] = None
    category_table_number: Optional[str] = None
    simulator_id: Optional[str] = None


class IntentPrediction(BaseModel):
    intent: Intent
    entities: IntentEntities = IntentEntities()