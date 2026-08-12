from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

SYSTEM_PROMPT = (
    "You are a lighting-standards assistant. Answer ONLY from the provided "
    "standard clauses. Cite the standard code, version year, ref number, table "
    "and page for every claim. If the clauses don't answer the question, say "
    "so clearly instead of guessing.\n\n"
    "ALL YOUR ANSWERS MUST BE IN MARKDOWN FORMAT (except intent detection)."
)

class CustomChatPromptTemplate:
    def __init__(self, question: str, citation: str | None, history: list[BaseMessage] | None, intent_prompt: str | None) -> None:
        self.question = question
        self.citation = citation
        self.history = history
        self.intent_prompt = intent_prompt


class RagService:
    def __init__(self, model: str, api_key: str):
        self.chat = ChatGroq(model=model, api_key=api_key, temperature=0.2)
        with_citations_msgs = [
            ("system", SYSTEM_PROMPT),
            ("system", "Context:\n{citations}"),
            ("system", "{intent_prompt}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
        no_citations_msgs = [
            ("system", SYSTEM_PROMPT),
            ("system", "There are no context just answer the question based on your knowledge and only if its a lighting related question otherwise say you don't know."),
            ("system", "{intent_prompt}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]

        self._chain_with_citations = ChatPromptTemplate.from_messages(with_citations_msgs) | self.chat | StrOutputParser()
        self._chain_no_citations = ChatPromptTemplate.from_messages(no_citations_msgs) | self.chat | StrOutputParser()

    def answer(self, template: CustomChatPromptTemplate) -> str:
        chain = self._chain_with_citations if template.citation else self._chain_no_citations
        if template.citation:
            return chain.invoke({
                "question": template.question,
                "citations": template.citation,
                "history": template.history or [],
                "intent_prompt": template.intent_prompt
            })
        else:
            return chain.invoke({
                "question": template.question,
                "history": template.history or [],
                "intent_prompt": template.intent_prompt
            })