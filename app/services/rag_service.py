from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

RAG_SYSTEM_PROMPT = (
    "You are a lighting-standards assistant. Answer ONLY from the provided "
    "standard clauses in the context. NEVER NEVER look up, recall, or invent "
    "requirements from outside that context (no memory of the standard, no "
    "general knowledge fill-ins). Cite the standard code, version year, ref "
    "number, table and page for every claim. If the clauses don't answer the "
    "question, say so clearly instead of guessing.\n\n"
    "ALL YOUR ANSWERS MUST BE IN MARKDOWN FORMAT (except intent detection)."
)

LLM_SYSTEM_PROMPT = (
    "You are a lighting-standards assistant focused on EN 12464-1 indoor "
    "workplace lighting. Answer from general lighting knowledge within the "
    "stated boundaries. Never invent regulatory numbers or fake citations. "
    "If you are unsure or the question needs clause lookup, say so.\n\n"
    "ALL YOUR ANSWERS MUST BE IN MARKDOWN FORMAT (except intent detection)."
)


class CustomChatPromptTemplate:
    def __init__(
        self,
        question: str,
        citation: str | None,
        history: list[BaseMessage] | None,
        intent_prompt: str | None = None,
        boundaries: str | None = None,
    ) -> None:
        self.question = question
        self.citation = citation
        self.history = history
        self.intent_prompt = intent_prompt
        self.boundaries = boundaries


class RagService:
    def __init__(self, model: str, api_key: str):
        self.chat = ChatGroq(model=model, api_key=api_key, temperature=0.2)
        self._parser = StrOutputParser()

    def answer(self, template: CustomChatPromptTemplate) -> str:
        messages: list = []
        if template.citation is not None:
            messages.append(("system", RAG_SYSTEM_PROMPT))
            context = template.citation or "(no matching clauses found)"
            messages.append(("system", f"Context:\n{context}"))
        else:
            messages.append(("system", LLM_SYSTEM_PROMPT))

        if template.boundaries:
            messages.append(("system", f"Boundaries:\n{template.boundaries}"))
        if template.intent_prompt:
            messages.append(("system", template.intent_prompt))

        messages.append(MessagesPlaceholder("history"))
        messages.append(("human", "{question}"))

        chain = ChatPromptTemplate.from_messages(messages) | self.chat | self._parser
        try:
            return chain.invoke({
                "question": template.question,
                "history": template.history or [],
            })
        except Exception as e:
            print("Error generating answer:", e)
            return (
                "I could not generate a full reply just now. Please try again, "
                "or rephrase the question."
            )
