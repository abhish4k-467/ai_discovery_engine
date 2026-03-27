import os
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

class VectorDB:
    def __init__(self, collection_name: str = "document-items"):
        self.vector_size = 384
        self.collection_name = collection_name
        qdrant_url = os.getenv("QDRANT_API_ENDPOINT")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            # Fallback to local file storage if no endpoint provided
            path=str("qdrant_data") if not qdrant_url else None,
        )
        self.embedding_model: Optional[TextEmbedding] = None
        self.embedding_error: Optional[str] = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _get_embedding_model(self) -> TextEmbedding:
        """Initializes embeddings lazily so API startup doesn't fail on first model download."""
        if self.embedding_model is not None:
            return self.embedding_model
        if self.embedding_error is not None:
            raise RuntimeError(self.embedding_error)

        try:
            self.embedding_model = TextEmbedding()
            return self.embedding_model
        except Exception as exc:
            self.embedding_error = (
                "Embedding model initialization failed. FastEmbed downloads models "
                "from Hugging Face on first run, so ensure network access or a preloaded cache. "
                f"Original error: {exc}"
            )
            raise RuntimeError(self.embedding_error) from exc

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _keyword_search_fallback(self, query: str, limit: int) -> List[Dict]:
        """Fallback retrieval when embedding-based search is unavailable."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        all_points = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if offset is None:
                break

        scored: List[Tuple[int, Dict]] = []
        for point in all_points:
            payload = point.payload or {}
            text = payload.get("text")
            if not text:
                continue

            score = len(query_tokens.intersection(self._tokenize(text)))
            if score > 0:
                scored.append((score, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "text": payload.get("text", ""),
                "source": payload.get("source"),
            }
            for _, payload in scored[:limit]
        ]

    def upsert(self, documents: List[Dict]) -> None:
        """Embeds and uploads documents to Qdrant."""
        if not documents:
            return

        texts = [doc["text"] for doc in documents]
        metadatas = [{"source": doc["source"], "text": doc["text"]} for doc in documents]

        # Generate embeddings
        try:
            embedding_model = self._get_embedding_model()
            vectors = [embedding.tolist() for embedding in embedding_model.embed(texts)]
        except RuntimeError as exc:
            print(f"Warning: {exc}")
            print("Falling back to zero vectors. Retrieval will use keyword matching.")
            vectors = [[0.0] * self.vector_size for _ in texts]

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=metadata,
            )
            for vector, metadata in zip(vectors, metadatas)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Upserted {len(points)} points to {self.collection_name}")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Searches for relevant documents."""
        try:
            embedding_model = self._get_embedding_model()
            query_vector = next(embedding_model.embed(query))

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=limit,
            )

            return [
                {"text": hit.payload["text"], "source": hit.payload.get("source")}
                for hit in results.points
                if hit.payload and hit.payload.get("text")
            ]
        except Exception:
            return self._keyword_search_fallback(query, limit)
