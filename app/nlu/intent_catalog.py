from dataclasses import dataclass

from app.nlu.intents import Intent, IntentCategory


@dataclass(frozen=True)
class IntentSpec:
    intent: Intent
    category: IntentCategory
    classifier_hint: str
    boundaries: str
    intent_prompt: str


INTENT_CATALOG: dict[Intent, IntentSpec] = {
    Intent.STANDARD_QUERY: IntentSpec(
        intent=Intent.STANDARD_QUERY,
        category=IntentCategory.RAG,
        classifier_hint=(
            "User asks about lighting levels/requirements for the current "
            "standard edition (no explicit year). If no standard is named, "
            "assume current EN 12464-1. Do not pick other standards unless "
            "explicitly named."
        ),
        boundaries=(
            "Answer ONLY from retrieved clauses. Prefer the current "
            "(is_latest) edition. Do not invent lux, UGR, Ra, or uniformity "
            "values. If retrieval is empty or insufficient, say the clauses "
            "were not found. Cite standard code, version year, ref number, "
            "table, and page for every claim. Respond in markdown."
        ),
        intent_prompt=(
            "NEVER NEVER look up, recall, or invent anything from outside the "
            "provided context. Use ONLY the retrieved clauses in this message. "
            "Answer the user's standards question from that context alone. Be "
            "precise and cite sources for each numeric or normative claim. If "
            "the context does not cover the question, say clearly that it was "
            "not found in the provided clauses — do not fill gaps from memory "
            "or general knowledge of the standard."
        ),
    ),
    Intent.HISTORICAL_QUERY: IntentSpec(
        intent=Intent.HISTORICAL_QUERY,
        category=IntentCategory.RAG,
        classifier_hint=(
            "User asks about lighting requirements for an explicit version "
            "year (e.g. '2019 version', 'EN 12464-1:2011'). Extract "
            "version_year."
        ),
        boundaries=(
            "Answer ONLY from retrieved clauses for the requested "
            "version_year. Never mix in current-edition values unless the "
            "user also asked for that. Cite code, year, ref, table, and page. "
            "If that year has no matching clauses, say so. Respond in markdown."
        ),
        intent_prompt=(
            "NEVER NEVER look up, recall, or invent anything from outside the "
            "provided context. Use ONLY the retrieved clauses in this message. "
            "Answer strictly for the historical edition identified in that "
            "context. Call out the version year. Do not substitute newer "
            "edition values or fill gaps from memory — if missing, say it was "
            "not found in the provided clauses."
        ),
    ),
    Intent.COMPARISON: IntentSpec(
        intent=Intent.COMPARISON,
        category=IntentCategory.RAG,
        classifier_hint=(
            "User asks to compare anything that needs standard clauses — "
            "editions/years, standards, OR categories/activities/tasks/"
            "space types within the same standard (e.g. 'compare 2011 vs "
            "2021 corridor illuminance', 'office vs corridor lighting "
            "levels', 'difference between classroom and meeting room "
            "requirements'). If no edition/standard is named, assume "
            "comparison within the current edition of the same standard "
            "(default EN 12464-1). Prefer comparison over general whenever "
            "two sides of normative requirements are being contrasted. "
            "Extract version_year and standard_code when mentioned."
        ),
        boundaries=(
            "Answer ONLY from retrieved clauses. Present each side with "
            "citations. Apply filters when present (edition/year, standard); "
            "if none are mentioned, compare categories/activities/tasks "
            "within the same current standard edition. Clearly state what "
            "differs vs what is the same. If one side is missing from "
            "context, say so instead of guessing. Respond in markdown."
        ),
        intent_prompt=(
            "NEVER NEVER look up, recall, or invent anything from outside the "
            "provided context. Use ONLY the retrieved clauses in this message. "
            "Compare the sides present in that context (editions and/or "
            "categories/activities/tasks). Cite every claim. Do not invent "
            "values for a missing side or complete the comparison from memory "
            "or general knowledge of the standard."
        ),
    ),
    Intent.REF_LOOKUP: IntentSpec(
        intent=Intent.REF_LOOKUP,
        category=IntentCategory.RAG,
        classifier_hint=(
            "User asks for a specific clause, reference number, table, or "
            "category (e.g. 'what does ref 5.3.2 say?', 'table 6.2 General "
            "areas', 'category 6.1'). Extract ref_number for clause ids "
            "(e.g. 6.2.1) and category_table_number for table/category ids "
            "(e.g. 6.2). If unsure whether it is a table or clause, fill both "
            "with the same value."
        ),
        boundaries=(
            "Answer ONLY from retrieved clauses. Anchor on the requested "
            "ref_number and/or category_table_number. Quote or closely "
            "paraphrase; do not expand into unrelated requirements. Cite "
            "code, year, ref, table, and page. Respond in markdown."
        ),
        intent_prompt=(
            "NEVER NEVER look up, recall, or invent anything from outside the "
            "provided context. Use ONLY the retrieved clauses in this message. "
            "Explain the specific clause/table/category from that context "
            "alone. Stay faithful to the wording; keep commentary minimal. If "
            "several activities share the same table, summarize only those "
            "present in the context. If the requested ref/table is not in the "
            "context, say it was not found — do not supply it from memory."
        ),
    ),
    Intent.GREETING: IntentSpec(
        intent=Intent.GREETING,
        category=IntentCategory.LLM,
        classifier_hint=(
            "Casual hello, thanks, goodbye, or short social message with no "
            "standards question."
        ),
        boundaries=(
            "No retrieval. No fake citations. No invented regulatory numbers. "
            "Keep the reply short and professional. Stay within a lighting-"
            "standards assistant persona."
        ),
        intent_prompt=(
            "Greet the user warmly and professionally. Thank them for using "
            "the lighting-standards assistant. Briefly invite a question about "
            "EN 12464-1 lighting requirements. Keep it concise."
        ),
    ),
    Intent.GENERAL: IntentSpec(
        intent=Intent.GENERAL,
        category=IntentCategory.LLM,
        classifier_hint=(
            "User asks for lighting definitions or concepts that can be "
            "answered without citing a specific standard clause (e.g. what is "
            "UGR, what is maintained illuminance). Do NOT use for comparing "
            "requirements between spaces, activities, categories, tasks, "
            "editions, or standards — those are comparison. Do NOT use when "
            "the user asks to open, show, use, or run a simulator or "
            "interactive visualization — that is get_simulator."
        ),
        boundaries=(
            "No retrieval and no fake citations. Explain concepts from general "
            "lighting knowledge. Do not invent normative lux/UGR/Ra limits. "
            "If the user needs numeric requirements from the standard, say they "
            "should ask about a specific space/edition so clauses can be "
            "retrieved. Refuse calculations, CAD, and non-lighting topics. "
            "Respond in markdown."
        ),
        intent_prompt=(
            "Explain the lighting concept clearly and accurately. If the "
            "question really needs EN 12464-1 numeric limits, say so and ask "
            "them to name the space or edition for a standards lookup. A "
            "matching interactive demo may be shown alongside this answer; "
            "do not invent links or iframe URLs."
        ),
    ),
    Intent.LIGHTING_GUIDANCE: IntentSpec(
        intent=Intent.LIGHTING_GUIDANCE,
        category=IntentCategory.LLM,
        classifier_hint=(
            "User wants practical lighting design advice or how-to guidance "
            "(qualitative), not a direct clause citation (e.g. tips for office "
            "lighting comfort). Do NOT use when the user asks to open, show, "
            "use, or run a simulator — that is get_simulator."
        ),
        boundaries=(
            "No retrieval. Qualitative guidance only. Never invent regulatory "
            "numbers or pretend to quote the standard. Explicitly note that "
            "advice is not a substitute for EN 12464-1. For numeric "
            "requirements, steer the user to ask a standards question. Refuse "
            "detailed photometric calculations and CAD. Respond in markdown."
        ),
        intent_prompt=(
            "Give practical, qualitative lighting guidance. State clearly that "
            "this is general advice, not a substitute for the standard. Offer "
            "to look up EN 12464-1 requirements if they name a space or edition. "
            "A matching interactive demo may be shown alongside this answer; "
            "do not invent links or iframe URLs."
        ),
    ),
    Intent.CLARIFY: IntentSpec(
        intent=Intent.CLARIFY,
        category=IntentCategory.LLM,
        classifier_hint=(
            "The question is about lighting/standards but is too ambiguous to "
            "answer safely (missing year, space type, which edition, or which "
            "ref). Prefer this over guessing."
        ),
        boundaries=(
            "Do not answer as if sure. Do not invent values or citations. Ask "
            "one focused clarifying question (year, space type, ref number, or "
            "edition). Optionally give 2 short example phrasings. Respond in "
            "markdown."
        ),
        intent_prompt=(
            "Ask one precise clarifying question so the user can be helped "
            "with EN 12464-1. Do not provide a full standards answer yet."
        ),
    ),
    Intent.OUT_OF_SCOPE: IntentSpec(
        intent=Intent.OUT_OF_SCOPE,
        category=IntentCategory.LLM,
        classifier_hint=(
            "Not about EN 12464-1 indoor workplace lighting levels — other "
            "standards (ISO, EN 12464-2, etc.), calculations, luminaire "
            "counts, CAD/DWG, or non-lighting topics. Interactive lighting "
            "simulators in the catalog (IES photometry viewer, school lighting "
            "demo, etc.) are in scope as get_simulator — do not decline those."
        ),
        boundaries=(
            "Politely decline. Do not attempt the out-of-scope task. Briefly "
            "say what you can help with: EN 12464-1 lighting requirements "
            "(illuminance, uniformity, glare/UGR, Ra). No fake citations. "
            "Respond in markdown."
        ),
        intent_prompt=(
            "Decline the request with a short reason. Redirect the user to "
            "EN 12464-1 lighting-requirement questions they can ask instead."
        ),
    ),
    Intent.FALLBACK: IntentSpec(
        intent=Intent.FALLBACK,
        category=IntentCategory.LLM,
        classifier_hint=(
            "Only when nothing else fits and the message is not clearly "
            "out_of_scope or clarify."
        ),
        boundaries=(
            "Admit uncertainty. Do not invent standards content. Propose 2–3 "
            "likely reinterpretations as short questions the user can confirm. "
            "Stay within EN 12464-1 lighting help. Respond in markdown."
        ),
        intent_prompt=(
            "Say you are not sure what they need. Offer two or three short "
            "example questions about EN 12464-1 lighting requirements they "
            "might mean."
        ),
    ),
    Intent.GET_SIMULATOR: IntentSpec(
        intent=Intent.GET_SIMULATOR,
        category=IntentCategory.API,
        classifier_hint=(
            "User wants to open, show, use, or run an interactive lighting "
            "simulator or visualization (IES file analyzer, polar curves, "
            "school lighting, CCT/CRI/flicker classroom demo, etc.). "
            "Prefer this over general when they want the tool itself, not a "
            "definition. Extract simulator_id when the catalog id is obvious "
            "(e.g. ies, school_lighting)."
        ),
        boundaries=(
            "Do not invent iframe URLs or query strings. The system attaches "
            "the matching simulator. Introduce the tool briefly. No fake "
            "citations. No CAD or luminaire-count calculations. Respond in "
            "markdown."
        ),
        intent_prompt=(
            "Introduce the interactive simulator briefly. Do not invent or "
            "paste iframe URLs or query strings. If the user mentioned values, "
            "you may refer to them in words. Keep it short."
        ),
    ),
}


def get_spec(intent: Intent | str) -> IntentSpec:
    if isinstance(intent, str):
        intent = Intent(intent)
    return INTENT_CATALOG[intent]


def classifier_prompt() -> str:
    rag_lines = []
    llm_lines = []
    api_lines = []
    for spec in INTENT_CATALOG.values():
        line = f"- {spec.intent.value}: {spec.classifier_hint}"
        if spec.category == IntentCategory.RAG:
            rag_lines.append(line)
        elif spec.category == IntentCategory.LLM:
            llm_lines.append(line)
        elif spec.category == IntentCategory.API:
            api_lines.append(line)

    return (
        "Classify the user message for a lighting-standards assistant "
        "focused on EN 12464-1 indoor workplace lighting.\n\n"
        "Categories:\n"
        "- RAG intents require searching standard clauses before answering.\n"
        "- LLM intents answer from bounded knowledge or conversation only "
        "(no clause search).\n"
        "- API intents resolve a catalog tool (simulator) and return a "
        "structured payload; they do not search clauses.\n\n"
        "RAG intents:\n"
        + "\n".join(rag_lines)
        + "\n\nLLM intents:\n"
        + "\n".join(llm_lines)
        + "\n\nAPI intents:\n"
        + "\n".join(api_lines)
        + "\n\nEntity extraction:\n"
        "- version_year: four-digit year only when explicitly mentioned "
        "(e.g. 2019).\n"
        "- standard_code: e.g. 'EN 12464-1' when named.\n"
        "- ref_number: clause/activity id like '6.2.1' when named.\n"
        "- category_table_number: table/category id like '6.2' or '5.28' "
        "when the user names a table or category (not a deep clause).\n"
        "- simulator_id: catalog id such as 'ies' or 'school_lighting' when "
        "the user clearly names that tool. Use null if unsure.\n"
        "Use null for any entity not present.\n\n"
        "Accept any language in the user message, but always return intent "
        "and entities in English."
    )
