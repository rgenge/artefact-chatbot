"""Gemini client and hybrid policy-manual retrieval."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .catalog import (
    QUERY_SYNONYMS,
    RetrievedChunk,
    keyword_set,
    keyword_tokens,
    normalize,
    query_model_numbers,
)
def load_dotenv(path: Path) -> None:
    """Small dependency-free .env loader, matching the project's simple CLI use."""

    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash").strip() or "gemini-3.1-flash"
GEMINI_EMBEDDING_MODEL = (
    os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    or "gemini-embedding-001"
)
GEMINI_EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768") or "768")
GOOGLE_API_KEY_ENV_NAMES = ("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY")


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    """Minimal REST client matching twintweaker's Gemini API calls."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = (self._read_key() if api_key is None else api_key).strip()
        self.model = model.strip() or GEMINI_MODEL

    @staticmethod
    def _read_key() -> str:
        for name in GOOGLE_API_KEY_ENV_NAMES:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post(self, model: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise GeminiError("Gemini API key not configured")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:{operation}"
        )
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=35) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GeminiError(f"Gemini {operation} failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GeminiError(f"Gemini {operation} connection failed: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GeminiError("Gemini returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise GeminiError("Gemini returned an invalid response")
        return data

    def generate(
        self,
        system_instruction: str,
        contents: list[dict[str, Any]],
        *,
        temperature: float = 0.35,
        max_output_tokens: int = 420,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.85,
                "maxOutputTokens": max_output_tokens,
            },
        }
        try:
            response = self._post(self.model, "generateContent", payload)
        except GeminiError as exc:
            print(f"[Gemini] {exc}", file=sys.stderr)
            return None
        candidates = response.get("candidates") or []
        if not candidates:
            return None
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if part.get("text"))
        return text.strip() or None

    def embed_documents(self, texts: list[str]) -> list[Optional[list[float]]]:
        if not self.enabled or not texts:
            return []
        embeddings: list[Optional[list[float]]] = []
        for start in range(0, len(texts), 50):
            batch = texts[start : start + 50]
            payload = {
                "requests": [
                    {
                        "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "outputDimensionality": GEMINI_EMBEDDING_DIMENSIONS,
                    }
                    for text in batch
                ]
            }
            try:
                data = self._post(GEMINI_EMBEDDING_MODEL, "batchEmbedContents", payload)
            except GeminiError as exc:
                print(f"[Gemini] document embeddings unavailable: {exc}", file=sys.stderr)
                return []
            embeddings.extend(
                self._normalize_embedding(item.get("values"))
                for item in data.get("embeddings", [])
            )
        return embeddings

    def embed_query(self, text: str) -> Optional[list[float]]:
        if not self.enabled:
            return None
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": GEMINI_EMBEDDING_DIMENSIONS,
        }
        try:
            data = self._post(GEMINI_EMBEDDING_MODEL, "embedContent", payload)
        except GeminiError as exc:
            print(f"[Gemini] query embedding unavailable: {exc}", file=sys.stderr)
            return None
        return self._normalize_embedding((data.get("embedding") or {}).get("values"))

    @staticmethod
    def _normalize_embedding(values: Any) -> Optional[list[float]]:
        if not isinstance(values, list) or not values:
            return None
        floats = [float(value) for value in values]
        magnitude = math.sqrt(sum(value * value for value in floats))
        if not magnitude:
            return floats
        return [value / magnitude for value in floats]


class PolicyRAG:
    """Hybrid PDF index: Gemini semantic retrieval plus twintweaker-style keywords."""

    TARGET_CHARS = 1400
    OVERLAP_CHARS = 180
    MAX_CHUNKS = 400
    MAX_RETRIEVED = 7
    VECTOR_THRESHOLD = 0.65
    KEYWORD_THRESHOLD = 4.0

    def __init__(self, pdf_path: str | Path, client: Optional[GeminiClient] = None):
        self.pdf_path = Path(pdf_path)
        self.client = client or GeminiClient()
        self.chunks = self._build_chunks()
        self.embeddings: list[Optional[list[float]]] = []
        if self.client.enabled:
            self.embeddings = self.client.embed_documents([chunk.content for chunk in self.chunks])

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\u00ad", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_chunks(self) -> list[RetrievedChunk]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Instale as dependências com: pip install -r requirements.txt") from exc

        chunks: list[RetrievedChunk] = []
        reader = PdfReader(str(self.pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean(page.extract_text() or "")
            if not text:
                continue
            start = 0
            while start < len(text) and len(chunks) < self.MAX_CHUNKS:
                content = text[start : start + self.TARGET_CHARS].strip()
                if len(content) >= 40:
                    chunks.append(
                        RetrievedChunk(
                            title=self.pdf_path.name,
                            source_type="policy_pdf",
                            content=content,
                            score=0.0,
                            retrieval="keyword",
                            page=page_number,
                        )
                    )
                if start + self.TARGET_CHARS >= len(text):
                    break
                start += self.TARGET_CHARS - self.OVERLAP_CHARS
        return chunks

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _keyword_score(query: str, content: str) -> float:
        query_words = keyword_set(query)
        content_words = keyword_set(content)
        if not query_words:
            return 0.0
        heading = normalize(content.splitlines()[0][:120] if content.splitlines() else "")
        lower_content = normalize(content)
        score = 0.0
        for word in query_words:
            score += 3.0 if word in content_words else 0.0
            score += 2.0 if word in heading else 0.0
            score += 1.0 if word in lower_content else 0.0
        expansion_words = set()
        for word in keyword_tokens(query):
            expansion_words.update(QUERY_SYNONYMS.get(word, set()))
        for word in expansion_words - query_words:
            score += 2.0 if word in content_words else 0.0
            score += 1.0 if word in lower_content else 0.0
        if normalize(query) in lower_content:
            score += 8.0
        return score

    def search(self, query: str, top_k: int = MAX_RETRIEVED) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        keyword_hits: list[RetrievedChunk] = []
        for chunk in self.chunks:
            score = self._keyword_score(query, chunk.content)
            if score >= self.KEYWORD_THRESHOLD:
                keyword_hits.append(
                    RetrievedChunk(
                        **{**chunk.__dict__, "score": score, "retrieval": "keyword"}
                    )
                )

        semantic_hits: list[RetrievedChunk] = []
        if self.embeddings:
            query_embedding = self.client.embed_query(query)
            if query_embedding:
                for chunk, embedding in zip(self.chunks, self.embeddings):
                    if embedding is None:
                        continue
                    score = self._cosine(query_embedding, embedding)
                    if score >= self.VECTOR_THRESHOLD:
                        semantic_hits.append(
                            RetrievedChunk(
                                **{**chunk.__dict__, "score": score, "retrieval": "semantic"}
                            )
                        )

        by_content: dict[str, RetrievedChunk] = {}
        keyword_first = self._prefer_keyword(query)
        for chunk in [*semantic_hits, *keyword_hits]:
            key = chunk.content[:120]
            existing = by_content.get(key)
            if existing is None:
                by_content[key] = chunk
                continue
            if keyword_first:
                if chunk.retrieval == "keyword" or chunk.score > existing.score:
                    by_content[key] = chunk
            elif chunk.retrieval == "semantic" or chunk.score > existing.score:
                by_content[key] = chunk

        ranked = list(by_content.values())
        ranked.sort(
            key=lambda chunk: (
                0 if keyword_first and chunk.retrieval == "keyword" else 1,
                -chunk.score,
            )
        )
        return ranked[:top_k]

    @staticmethod
    def _prefer_keyword(query: str) -> bool:
        words = keyword_set(query)
        business_terms = {
            "preco", "custa", "valor", "estoque", "disponivel", "pagamento",
            "pix", "troca", "devolucao", "garantia", "defeito", "entrega",
            "frete", "horario", "endereco", "pedido", "rastreamento",
        }
        return bool(query_model_numbers(query) or words & business_terms)



__all__ = [
    "GEMINI_EMBEDDING_DIMENSIONS",
    "GEMINI_EMBEDDING_MODEL",
    "GEMINI_MODEL",
    "GeminiClient",
    "GeminiError",
    "PolicyRAG",
    "load_dotenv",
]