from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List


class StandardMetadata(BaseModel):
    standard_code: str
    version_year: str
    is_latest: bool


class Hierarchy(BaseModel):
    category_table_number: str
    category_title: str
    ref_number: str
    page: int


class Parameters(BaseModel):
    em_r_lx: Optional[float] = None
    em_u_lx: Optional[float] = None
    uo: Optional[float] = None
    ra: Optional[float] = None
    ugr_rugl: Optional[float] = None
    ez_lx: Optional[float] = None
    em_wall_lx: Optional[float] = None
    em_ceiling_lx: Optional[float] = None


class ClauseDocument(BaseModel):
    """The standard clause document shape (single source of truth for both
    /ingest and /documents). Accepts the wire field `_id`, exposes `id`."""

    id: str = Field(validation_alias="_id", serialization_alias="_id")
    qdrant_point_id: Optional[int] = None
    standard_metadata: StandardMetadata
    hierarchy: Hierarchy
    activity: str
    parameters: Parameters
    specific_requirements: Optional[str] = None
    searchable_text: str

    model_config = ConfigDict(populate_by_name=True)


class IngestRequest(BaseModel):
    documents: List[ClauseDocument]


class IngestResultItem(BaseModel):
    mongo_id: str
    status: str
    error: Optional[str] = None


class IngestResponse(BaseModel):
    results: List[IngestResultItem]