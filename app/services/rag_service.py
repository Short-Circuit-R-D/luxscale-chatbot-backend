from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

SYSTEM_PROMPT = (
    "You are a lighting-standards assistant. Answer ONLY from the provided "
    "standard clauses. Cite the standard code, version year, ref number, table "
    "and page for every claim. If the clauses don't answer the question, say "
    "so clearly instead of guessing."
)


class RagService:
    def __init__(self, model: str, api_key: str):
        self.chat = ChatGroq(model=model, api_key=api_key, temperature=0.2)
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("system", "Relevant standard clauses:\n{citations}"),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ])
            | self.chat
            | StrOutputParser()
        )

    def answer(self, question: str, citations: str, history: list[BaseMessage] | None = None) -> str:
        return self._chain.invoke({
            "question": question,
            "citations": citations or "No clauses retrieved.",
            "history": history or [],
        })