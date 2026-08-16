from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self._lc = None

    def _load(self) -> HuggingFaceEmbeddings:
        if self._lc is None:
            self._lc = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
            )
        return self._lc

    def load(self) -> None:
        """Eagerly load the model into memory (call at process startup)."""
        self._load()

    @property
    def lc_embeddings(self) -> HuggingFaceEmbeddings:
        return self._load()

    @property
    def loaded(self) -> bool:
        return self._lc is not None

    def embed(self, text: str) -> list[float]:
        vector = self._load().embed_query(text)
        return list(vector)