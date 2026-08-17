from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "simulators.json"

ParamType = Literal["number", "integer", "string"]


class SimulatorParamSpec(BaseModel):
    key: str
    type: ParamType = "string"
    description: str = ""
    positive: bool = False
    enum: list[str] = Field(default_factory=list)
    enum_aliases: dict[str, str] = Field(default_factory=dict)


class SimulatorSpec(BaseModel):
    id: str
    title: str
    iframe_base_url: str
    aliases: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    params: list[SimulatorParamSpec] = Field(default_factory=list)


class SimulatorCatalog(BaseModel):
    simulators: list[SimulatorSpec]


class ExtractedParam(BaseModel):
    key: str
    value: Optional[str] = None


class ExtractedParams(BaseModel):
    params: list[ExtractedParam] = Field(default_factory=list)


@dataclass(frozen=True)
class SimulatorPayload:
    id: str
    title: str
    iframe_url: str

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "iframe_url": self.iframe_url}


@dataclass(frozen=True)
class SimulatorMatch:
    spec: SimulatorSpec
    score: float


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_phrase(query_norm: str, phrase: str) -> bool:
    p = _normalize(phrase)
    if not p:
        return False
    return re.search(rf"(^|\s){re.escape(p)}(\s|$)", query_norm) is not None


def _phrase_score(query_norm: str, phrases: list[str]) -> float:
    best = 0.0
    for phrase in phrases:
        p = _normalize(phrase)
        if not p or not _contains_phrase(query_norm, phrase):
            continue
        best = max(best, 1.0 + (len(p) / 100.0))
    return best


def _coerce_enum(value: str, spec: SimulatorParamSpec) -> str | None:
    normalized = _normalize(value)
    if not normalized:
        return None
    for alias, canonical in spec.enum_aliases.items():
        if _normalize(alias) == normalized:
            return canonical
    for allowed in spec.enum:
        if _normalize(allowed) == normalized:
            return allowed
    return None


def _coerce_param(value: str, spec: SimulatorParamSpec) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    if spec.enum:
        return _coerce_enum(raw, spec)
    if spec.type == "number":
        try:
            number = float(raw.replace(",", ""))
        except ValueError:
            return None
        if spec.positive and number <= 0:
            return None
        if number.is_integer():
            return str(int(number))
        return str(number)
    if spec.type == "integer":
        try:
            number = int(float(raw.replace(",", "")))
        except ValueError:
            return None
        if spec.positive and number <= 0:
            return None
        return str(number)
    return raw


def _base_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class SimulatorService:
    def __init__(self, chat: ChatGroq, catalog_path: Path | None = None):
        path = catalog_path or CATALOG_PATH
        raw = json.loads(path.read_text(encoding="utf-8"))
        catalog = SimulatorCatalog.model_validate(raw)
        self._specs = list(catalog.simulators)
        self._by_id = {spec.id.lower(): spec for spec in self._specs}
        self._llm = chat.with_structured_output(
            ExtractedParams, method="json_mode"
        )

    def resolve_for_direct(
        self,
        query: str,
        simulator_id: str | None = None,
    ) -> tuple[SimulatorPayload | None, list[SimulatorSpec]]:
        """Return (payload, ambiguous_or_empty_candidates).

        payload is set on a unique match. candidates are listed when none or
        several catalog entries match so the handler can clarify.
        """
        spec = self._spec_from_id(simulator_id)
        if spec is None:
            matches = self._score(query, phrases_of=lambda s: [s.id, s.title, *s.aliases])
            strong = [m for m in matches if m.score > 0]
            if not strong:
                return None, list(self._specs)
            top = strong[0].score
            tied = [m.spec for m in strong if m.score == top]
            if len(tied) > 1:
                return None, tied
            spec = tied[0]
        payload = self._build(spec, query)
        return payload, []

    def resolve_for_concept(self, query: str) -> SimulatorPayload | None:
        matches = self._score(query, phrases_of=lambda s: s.concepts)
        strong = [m for m in matches if m.score > 0]
        if len(strong) != 1:
            return None
        return self._build(strong[0].spec, query)

    def _spec_from_id(self, simulator_id: str | None) -> SimulatorSpec | None:
        if not simulator_id:
            return None
        key = simulator_id.strip().lower()
        if key in self._by_id:
            return self._by_id[key]
        for spec in self._specs:
            names = [spec.id, spec.title, *spec.aliases]
            if any(_normalize(name) == _normalize(key) for name in names if name):
                return spec
        return None

    def _score(self, query: str, phrases_of) -> list[SimulatorMatch]:
        query_norm = _normalize(query)
        matches = [
            SimulatorMatch(spec=spec, score=_phrase_score(query_norm, phrases_of(spec)))
            for spec in self._specs
        ]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _build(self, spec: SimulatorSpec, query: str) -> SimulatorPayload:
        values = self._extract_params(spec, query)
        iframe_url = _base_url(spec.iframe_base_url)
        if values:
            iframe_url = f"{iframe_url}?{urlencode(values)}"
        return SimulatorPayload(id=spec.id, title=spec.title, iframe_url=iframe_url)

    def _extract_params(self, spec: SimulatorSpec, query: str) -> dict[str, str]:
        if not spec.params:
            return {}
        allowed = {p.key: p for p in spec.params}
        lines = "\n".join(
            f"- {p.key} ({p.type}): {p.description}"
            + (f" Allowed values: {', '.join(p.enum)}." if p.enum else "")
            for p in spec.params
        )
        prompt = (
            "Extract simulator input values mentioned in the user message. "
            "Only use these keys:\n"
            f"{lines}\n"
            "Return them as `params` objects with `key` and `value`. "
            "Omit keys that are not clearly present. Do not invent values.\n\n"
            f"User message: {query}"
        )
        try:
            extracted = self._llm.invoke(prompt)
        except Exception as e:
            print("Error extracting simulator params:", e)
            return {}

        cleaned: dict[str, str] = {}
        for item in extracted.params or []:
            spec_param = allowed.get(item.key)
            if spec_param is None or item.value is None:
                continue
            coerced = _coerce_param(str(item.value), spec_param)
            if coerced is not None:
                cleaned[item.key] = coerced
        return cleaned
