"""Empório da Música - hybrid Gemini customer-service agent.

This follows the retrieval split used by twintweaker:

* CSV catalog/order facts use deterministic structured search. They are never
  guessed from a document embedding.
* The policy PDF is chunked, embedded with Gemini's retrieval embedding model,
  and searched with semantic + keyword retrieval. Keyword retrieval remains a
  safe fallback when no API key is configured.
* Gemini 3.1 Flash writes the final short response from the retrieved context.

Run "python app.py" from this directory.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


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


load_dotenv(Path(__file__).resolve().with_name(".env"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash").strip() or "gemini-3.1-flash"
GEMINI_EMBEDDING_MODEL = (
    os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    or "gemini-embedding-001"
)
GEMINI_EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768") or "768")
GOOGLE_API_KEY_ENV_NAMES = ("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY")

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "ha", "me", "na", "nas", "no", "nos", "o", "os",
    "para", "por", "qual", "quais", "quanto", "que", "se", "sem", "tem",
    "temos", "ter", "um", "uma", "umas", "uns", "voce", "voces", "eu",
}

QUERY_SYNONYMS: dict[str, set[str]] = {
    "troca": {"devolucao", "return", "exchange"},
    "trocar": {"devolucao", "return", "exchange"},
    "devolucao": {"troca", "return", "exchange"},
    "devolver": {"devolucao", "arrependimento"},
    "arrependimento": {"devolucao", "devolver"},
    "garantia": {"defeito", "warranty"},
    "defeito": {"garantia", "warranty"},
    "warranty": {"garantia"},
    "entrega": {"envio", "delivery"},
    "envio": {"entrega", "delivery"},
    "frete": {"entrega", "envio"},
    "rastrear": {"rastreamento", "codigo"},
    "rastreamento": {"rastrear", "codigo"},
    "horario": {"expediente", "funcionamento", "aberto"},
    "endereco": {"localizacao"},
    "pix": {"pagamento"},
    "pagamento": {"pix", "boleto", "cartao"},
    "cabo": {"cabos"},
    "corda": {"cordas"},
    "palheta": {"palhetas"},
    "acessorio": {"acessorios"},
    "oferta": {"promocao", "desconto"},
    "ofertas": {"promocoes", "descontos"},
    "especial": {"promocao", "desconto"},
    "especiais": {"promocoes", "descontos"},
    "exclusivo": {"promocao", "desconto"},
    "exclusiva": {"promocao", "desconto"},
    "black": {"promocao", "promocoes"},
    "friday": {"promocao", "promocoes"},
}

ACCESSORY_TERMS = {
    "acessorio", "acessorios", "amplificador", "amplificadores", "cabo",
    "cabos", "case", "cases", "corda", "cordas", "palheta", "palhetas",
    "pedal", "pedais",
}

CATEGORY_ALIASES = {
    "violao": "Violões",
    "violoes": "Violões",
    "guitarra": "Guitarras",
    "guitarras": "Guitarras",
    "baixo": "Baixos",
    "baixos": "Baixos",
    "bateria": "Baterias e Percussão",
    "baterias": "Baterias e Percussão",
    "percussao": "Baterias e Percussão",
    "teclado": "Teclados e Pianos",
    "teclados": "Teclados e Pianos",
    "piano": "Teclados e Pianos",
    "pianos": "Teclados e Pianos",
    "sopro": "Instrumentos de Sopro",
    "ukulele": "Ukuleles",
    "ukuleles": "Ukuleles",
    "guitar": "Guitarras",
    "guitars": "Guitarras",
    "bass": "Baixos",
    "drum": "Baterias e Percussão",
    "drums": "Baterias e Percussão",
}

STATUS_LABELS = {
    "pending": "aguardando processamento",
    "confirmed": "confirmado",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split())


def keyword_tokens(value: Any) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", normalize(value))
    result: set[str] = {word for word in raw if word not in STOPWORDS and len(word) >= 2}
    for word in tuple(result):
        result.update(QUERY_SYNONYMS.get(word, set()))
    for word in tuple(result):
        if len(word) > 4 and word.endswith("oes"):
            result.add(word[:-3] + "ao")
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("is", "us")):
            result.add(word[:-1])
    result.update(
        token
        for token in raw
        if re.fullmatch(r"(?:[a-z]{1,4}\d{1,4}|\d{2,4})(?:gb|tb)?", token)
    )
    return list(result)


def keyword_set(value: Any) -> set[str]:
    return set(keyword_tokens(value))


def query_model_numbers(value: str) -> list[str]:
    result: list[str] = []
    for token in re.findall(r"[a-z0-9]+", normalize(value)):
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        if re.fullmatch(r"\d{2,4}", token) or re.fullmatch(r"[a-z]\d{1,3}", token):
            result.append(token)
        elif re.fullmatch(r"\d{1,4}(?:gb|tb)", token):
            result.append(token)
    return list(dict.fromkeys(result))


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = re.sub(r"[^0-9,.-]", "", str(value).strip())
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1 or (
        text.count(".") == 1 and len(text.rsplit(".", 1)[1]) == 3
    ):
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any, default: int = 0) -> int:
    number = parse_number(value)
    return int(number) if number is not None else default


def format_brl(value: float) -> str:
    formatted = f"R$ {value:,.2f}"
    return formatted.replace(",", "#").replace(".", ",").replace("#", ".")


def parse_budget(message: str) -> Optional[float]:
    match = re.search(
        r"(?:até|ate|menos de|abaixo de|no máximo|no maximo|por até|por ate)"
        r"\s*(?:r\$)?\s*([\d.]+(?:,[\d]{1,2})?)",
        message,
        re.IGNORECASE,
    )
    return parse_number(match.group(1)) if match else None


def parse_specs(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def locate_csv(data_dir: Path, stem: str) -> Path:
    matches = sorted(data_dir.glob(f"*{stem}*.csv"))
    if not matches:
        raise FileNotFoundError(f"CSV não encontrado para: {stem}")
    return matches[0]


@dataclass(frozen=True)
class RetrievedChunk:
    title: str
    source_type: str
    content: str
    score: float
    retrieval: str
    page: Optional[int] = None


@dataclass(frozen=True)
class Product:
    product_id: int
    price_brl: float
    name: str
    category_id: int
    description: str
    stock_quantity: int
    status: str
    specs: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.status == "active" and self.stock_quantity > 0


@dataclass(frozen=True)
class Promotion:
    promotion_id: int
    product_id: int
    discount_percent: float
    description: str
    is_active: bool


@dataclass(frozen=True)
class Customer:
    customer_id: int
    name: str
    phone: str
    email: str
    city: str


@dataclass(frozen=True)
class Order:
    order_id: int
    customer_id: int
    order_date: str
    status: str
    total_brl: float
    payment_method: str
    tracking_code: str
    estimated_delivery: str
    notes: str


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


class CatalogStore:
    """Structured catalog/order access; equivalent to twintweaker source_rows."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.products = self._load_products()
        self.promotions = self._load_promotions()
        self.categories = self._load_categories()
        self.products_by_id = {item.product_id: item for item in self.products}
        # These tables are loaded only when a customer/order flow needs them.
        self._customers_cache = None
        self._orders_cache = None
        self._order_items_cache = None
        self._customers_by_id_cache = None
        self._orders_by_id_cache = None
        self._orders_by_customer_cache = None
        self.promotions_by_product: dict[int, list[Promotion]] = defaultdict(list)
        for promotion in self.promotions:
            self.promotions_by_product[promotion.product_id].append(promotion)
        self.categories.setdefault(9, "Ukuleles")

    @property
    def customers(self) -> list[Customer]:
        if self._customers_cache is None:
            self._customers_cache = self._load_customers()
        return self._customers_cache

    @property
    def orders(self) -> list[Order]:
        if self._orders_cache is None:
            self._orders_cache = self._load_orders()
        return self._orders_cache

    @property
    def order_items(self) -> dict[int, list[dict[str, int]]]:
        if self._order_items_cache is None:
            self._order_items_cache = self._load_order_items()
        return self._order_items_cache

    @property
    def customers_by_id(self) -> dict[int, Customer]:
        if self._customers_by_id_cache is None:
            self._customers_by_id_cache = {item.customer_id: item for item in self.customers}
        return self._customers_by_id_cache

    @property
    def orders_by_id(self) -> dict[int, Order]:
        if self._orders_by_id_cache is None:
            self._orders_by_id_cache = {item.order_id: item for item in self.orders}
        return self._orders_by_id_cache

    @property
    def orders_by_customer(self) -> dict[int, list[Order]]:
        if self._orders_by_customer_cache is None:
            result = defaultdict(list)
            for order in self.orders:
                result[order.customer_id].append(order)
            for customer_orders in result.values():
                customer_orders.sort(key=lambda item: item.order_date, reverse=True)
            self._orders_by_customer_cache = result
        return self._orders_by_customer_cache
    def _load_products(self) -> list[Product]:
        return [
            Product(
                product_id=parse_int(row.get("product_id")),
                price_brl=parse_number(row.get("price_brl")) or 0.0,
                name=row.get("name", "").strip(),
                category_id=parse_int(row.get("category_id")),
                description=row.get("description", "").strip(),
                stock_quantity=parse_int(row.get("stock_quantity")),
                status=row.get("status", "").strip().lower(),
                specs=parse_specs(row.get("specs", "")),
            )
            for row in read_csv(locate_csv(self.data_dir, "products"))
        ]

    def _load_promotions(self) -> list[Promotion]:
        return [
            Promotion(
                promotion_id=parse_int(row.get("promotion_id")),
                product_id=parse_int(row.get("product_id")),
                discount_percent=parse_number(row.get("discount_percent")) or 0.0,
                description=row.get("description", "").strip(),
                is_active=normalize(row.get("is_active")) in {"1", "true", "yes", "sim"},
            )
            for row in read_csv(locate_csv(self.data_dir, "promotions"))
        ]

    def _load_customers(self) -> list[Customer]:
        return [
            Customer(
                customer_id=parse_int(row.get("customer_id")),
                name=row.get("name", "").strip(),
                phone=row.get("phone", "").strip(),
                email=row.get("email", "").strip(),
                city=row.get("city", "").strip(),
            )
            for row in read_csv(locate_csv(self.data_dir, "customers"))
        ]

    def _load_orders(self) -> list[Order]:
        return [
            Order(
                order_id=parse_int(row.get("order_id")),
                customer_id=parse_int(row.get("customer_id")),
                order_date=row.get("order_date", "").strip(),
                status=row.get("status", "").strip().lower(),
                total_brl=parse_number(row.get("total_brl")) or 0.0,
                payment_method=row.get("payment_method", "").strip(),
                tracking_code=row.get("tracking_code", "").strip(),
                estimated_delivery=row.get("estimated_delivery", "").strip(),
                notes=row.get("notes", "").strip(),
            )
            for row in read_csv(locate_csv(self.data_dir, "orders"))
        ]

    def _load_order_items(self) -> dict[int, list[dict[str, int]]]:
        result: dict[int, list[dict[str, int]]] = defaultdict(list)
        for row in read_csv(locate_csv(self.data_dir, "order_items")):
            result[parse_int(row.get("order_id"))].append(
                {
                    "quantity": parse_int(row.get("quantity")),
                    "product_id": parse_int(row.get("product_id")),
                }
            )
        return result

    def _load_categories(self) -> dict[int, str]:
        return {
            parse_int(row.get("category_id")): row.get("name", "").strip()
            for row in read_csv(locate_csv(self.data_dir, "categories"))
        }

    def category_name(self, category_id: int) -> str:
        return self.categories.get(category_id, "Instrumentos musicais")

    def category_id_for(self, query: str) -> Optional[int]:
        normalized = normalize(query)
        for alias, category in CATEGORY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}s?\b", normalized):
                for category_id, name in self.categories.items():
                    if normalize(name) == normalize(category):
                        return category_id
                if category == "Ukuleles":
                    return 9
        return None

    def active_promotion(self, product: Product) -> Optional[Promotion]:
        if not product.available:
            return None
        return max(
            (item for item in self.promotions_by_product.get(product.product_id, []) if item.is_active),
            key=lambda item: item.discount_percent,
            default=None,
        )

    def effective_price(self, product: Product) -> float:
        promotion = self.active_promotion(product)
        return round(product.price_brl * (1 - promotion.discount_percent / 100), 2) if promotion else product.price_brl

    def pix_price(self, product: Product) -> Optional[float]:
        return None if self.active_promotion(product) else round(product.price_brl * 0.95, 2)

    def _name_score(self, product: Product, query: str) -> float:
        query_words = keyword_set(query)
        name_words = keyword_set(product.name)
        searchable = keyword_set(f"{product.name} {product.description} {self.category_name(product.category_id)}")
        score = 5.0 * len(query_words & name_words) + 1.5 * len(query_words & searchable)
        if normalize(product.name) in normalize(query):
            score += 20.0
        if normalize(self.category_name(product.category_id)) in normalize(query):
            score += 6.0
        return score

    def search_products(
        self,
        query: str,
        *,
        category_id: Optional[int] = None,
        max_price: Optional[float] = None,
        include_unavailable: bool = False,
        limit: int = 5,
    ) -> list[Product]:
        candidates: list[tuple[float, float, Product]] = []
        for product in self.products:
            if category_id is not None and product.category_id != category_id:
                continue
            if not include_unavailable and not product.available:
                continue
            price = self.effective_price(product) if product.available else product.price_brl
            if max_price is not None and price > max_price:
                continue
            candidates.append((self._name_score(product, query), price, product))
        browse = bool(category_id is not None or max_price is not None)
        candidates.sort(
            key=lambda item: (item[1], -item[0], normalize(item[2].name))
            if browse else (-item[0], item[1], normalize(item[2].name))
        )
        return [item[2] for item in candidates[:limit]]

    def best_product_match(self, query: str) -> Optional[Product]:
        query_norm = normalize(query)
        exact = [item for item in self.products if normalize(item.name) in query_norm]
        if exact:
            return exact[0]
        scored = sorted(
            ((self._name_score(item, query), item) for item in self.products),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            return None
        best_score, product = scored[0]
        if best_score < 8.0:
            return None
        hits = len(keyword_set(query) & keyword_set(product.name))
        return product if hits >= 2 else None

    def brand_products(
        self,
        query: str,
        *,
        category_id: Optional[int] = None,
        max_price: Optional[float] = None,
        include_unavailable: bool = False,
    ) -> list[Product]:
        """Return every product matching a brand term, without the normal top-five cap."""
        query_words = keyword_set(query)
        brand_terms = {
            normalize(product.name).split()[0]
            for product in self.products
            if normalize(product.name).split()
        }
        requested_brands = query_words & brand_terms
        if not requested_brands:
            return []
        matches = []
        for product in self.products:
            name_words = normalize(product.name).split()
            if not name_words or name_words[0] not in requested_brands:
                continue
            if category_id is not None and product.category_id != category_id:
                continue
            if not include_unavailable and not product.available:
                continue
            price = self.effective_price(product) if product.available else product.price_brl
            if max_price is not None and price > max_price:
                continue
            matches.append(product)
        return sorted(matches, key=lambda item: (self.effective_price(item), normalize(item.name)))

    def similar_products(self, product: Product, limit: int = 3) -> list[Product]:
        items = [
            item for item in self.products
            if item.product_id != product.product_id
            and item.category_id == product.category_id
            and item.available
        ]
        return sorted(
            items,
            key=lambda item: (abs(self.effective_price(item) - product.price_brl), item.name),
        )[:limit]

    def active_promotions(self, limit: int = 5) -> list[tuple[Product, Promotion]]:
        result = []
        for product in self.products:
            promotion = self.active_promotion(product)
            if promotion:
                result.append((product, promotion))
        result.sort(key=lambda pair: (self.effective_price(pair[0]), pair[0].name))
        return result[:limit]

    def identify_customer(self, message: str) -> Optional[Customer]:
        message_norm = normalize(message)
        digits = re.sub(r"\D", "", message)
        for customer in self.customers:
            if normalize(customer.email) in message_norm or normalize(customer.name) in message_norm:
                return customer
            phone = re.sub(r"\D", "", customer.phone)
            if phone and phone in digits:
                return customer
        return None

    def order_by_reference(self, reference: Optional[str]) -> Optional[Order]:
        if not reference:
            return None
        value = reference.strip().upper()
        if value.isdigit():
            return self.orders_by_id.get(int(value))
        return next((order for order in self.orders if order.tracking_code.upper() == value), None)

    def latest_order(self, customer_id: int) -> Optional[Order]:
        return (self.orders_by_customer.get(customer_id) or [None])[0]

    def order_items_for(self, order_id: int) -> list[tuple[int, str, int]]:
        result = []
        for item in self.order_items.get(order_id, []):
            product = self.products_by_id.get(item["product_id"])
            result.append((item["product_id"], product.name if product else "Produto", item["quantity"]))
        return result

    def search_grounding(
        self,
        query: str,
        *,
        category_id: Optional[int] = None,
        max_price: Optional[float] = None,
        include_unavailable: bool = False,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        products = self.search_products(
            query,
            category_id=category_id,
            max_price=max_price,
            include_unavailable=include_unavailable,
            limit=limit,
        )
        chunks: list[RetrievedChunk] = []
        for product in products:
            promotion = self.active_promotion(product)
            price = self.effective_price(product) if product.available else product.price_brl
            details = [
                f"Nome: {product.name}",
                f"Categoria: {self.category_name(product.category_id)}",
                f"Preço: {format_brl(price)}",
                f"Preço original: {format_brl(product.price_brl)}",
                f"Estoque: {product.stock_quantity}",
                f"Status: {product.status}",
                f"Descrição: {product.description}",
            ]
            if promotion:
                details.append(
                    f"Promoção ativa: {promotion.discount_percent:g}% ({promotion.description})"
                )
            chunks.append(
                RetrievedChunk(
                    title=product.name,
                    source_type="catalog_row",
                    content="\n".join(details),
                    score=10.0,
                    retrieval="keyword",
                )
            )
        return chunks


class StoreAgent:
    """Conversation state, retrieval orchestration and grounded Gemini response."""

    SYSTEM_PROMPT = """Você representa oficialmente a Empório da Música, em Campo Grande/MS.
Fale como um membro da equipe, em português do Brasil, com tom acolhedor, informal
e profissional. Seja breve, normalmente 2 a 5 frases ou uma lista curta.

Escopo: instrumentos musicais, catálogo, preços, estoque, pedidos, entregas e
políticas da loja. Redirecione pedidos de acessórios e assuntos fora do escopo.
Use somente o contexto confiável fornecido nesta mensagem e o histórico. Nunca
invente preço, estoque, promoção, prazo, status, produto, variante ou regra.
Se a informação não estiver no contexto, diga que precisa confirmar com a equipe.
Não mencione IA, modelo, prompt, RAG, embeddings ou fontes internas. Não exponha
dados pessoais além do necessário para responder ao próprio cliente.
"""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        customer_id: Optional[int] = None,
        use_llm: bool = True,
        model: Optional[str] = None,
    ):
        data_path = Path(data_dir)
        self.store = CatalogStore(data_path)
        self.client = GeminiClient(api_key=None if use_llm else "", model=model or GEMINI_MODEL)
        self.rag = PolicyRAG(data_path / "políticas_da_loja.pdf", self.client)
        self.customer_id = customer_id
        self.history: list[dict[str, str]] = []
        self.use_llm = use_llm

    @property
    def customer(self) -> Optional[Customer]:
        return self.store.customers_by_id.get(self.customer_id) if self.customer_id else None

    def _remember_customer(self, message: str) -> None:
        # Identity lookup is opt-in; catalog/promotion questions do not load customers.
        if self.customer_id is not None:
            return
        if not re.search(r"@|\+?\d[\d ()-]{7,}|\b(?:sou|meu nome|meu email)\b", message, re.IGNORECASE):
            return
        customer = self.store.identify_customer(message)
        if customer:
            self.customer_id = customer.customer_id
    def handle(self, message: str) -> str:
        message = message.strip()
        if not message:
            return "Pode me contar como posso ajudar?"
        self._remember_customer(message)
        catalog_message = message
        if re.search(r"\b(?:todos|todas|cada|lista completa|mais modelos)\b", normalize(message)) and self.history:
            previous_user = next((item["content"] for item in reversed(self.history) if item["role"] == "user"), "")
            if previous_user and route_message(previous_user, self.store) == "catalog":
                catalog_message = previous_user + " " + message
        intent = route_message(catalog_message, self.store)

        if intent == "greeting":
            answer = self._greeting()
            grounding: list[RetrievedChunk] = []
        elif intent == "out_of_scope":
            answer = self._out_of_scope()
            grounding = []
        elif intent == "catalog":
            answer, grounding = self._catalog_answer(catalog_message)
        elif intent == "order":
            answer, grounding = self._order_answer(message)
        elif intent == "policy":
            answer, grounding = self._policy_answer(message)
        else:
            answer, grounding = self._unknown(), []

        if self.use_llm and self.client.enabled and intent in {"catalog", "order", "policy", "unknown"}:
            generated = self._gemini_answer(message, grounding)
            if generated and self._response_is_grounded(generated, grounding):
                answer = generated

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _gemini_answer(self, message: str, grounding: list[RetrievedChunk]) -> Optional[str]:
        source_block = (
            "\n\n---\n\n".join(
                f"SOURCE {index}: {chunk.title} ({chunk.source_type})\n{chunk.content}"
                for index, chunk in enumerate(grounding, start=1)
            )
            if grounding
            else "No relevant source chunks were found for this message."
        )
        history_block = "\n".join(
            f"{item['role']}: {item['content']}" for item in self.history[-10:]
        ) or "No recent conversation history."
        list_all_instruction = (
            "Quando o cliente pedir todos, quais modelos ou a lista completa, enumere "
            "todos os itens correspondentes presentes no contexto confiável; não reduza a resposta "
            "ao primeiro resultado nem repita uma resposta anterior."
            if re.search(r"\b(?:todos|todas|quais|lista completa|mais modelos)\b", normalize(message))
            else ""
        )
        system = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"{list_all_instruction}\n\n"
            "Trusted context:\n"
            f"{source_block}\n\n"
            "Recent conversation:\n"
            f"{history_block}"
        )
        contents = [
            {"role": "model" if item["role"] == "assistant" else "user", "parts": [{"text": item["content"]}]}
            for item in self.history[-10:]
        ]
        contents.append({"role": "user", "parts": [{"text": message}]})
        return self.client.generate(system, contents)

    @staticmethod
    def _response_is_grounded(answer: str, grounding: list[RetrievedChunk]) -> bool:
        mentioned = re.findall(r"r\$\s*[\d.,]+", normalize(answer))
        if not mentioned:
            return True
        trusted = normalize("\n".join(chunk.content for chunk in grounding))
        trusted_digits = re.sub(r"[^0-9]", "", trusted)
        for value in mentioned:
            number = re.sub(r"[^0-9]", "", value)
            if number and number not in trusted_digits:
                return False
        return True

    def _greeting(self) -> str:
        name = f" {self.customer.name.split()[0]}" if self.customer else ""
        return (
            f"Oi{name}! Eu sou da Empório da Música. Posso ajudar com instrumentos, "
            "preços, estoque, pedidos e políticas da loja. O que você procura?"
        )

    @staticmethod
    def _out_of_scope() -> str:
        return (
            "A Empório da Música trabalha exclusivamente com instrumentos musicais; não vendemos "
            "cordas, cabos, palhetas, cases, pedais ou amplificadores. Posso ajudar com instrumentos."
        )

    @staticmethod
    def _unknown() -> str:
        return (
            "Posso ajudar com instrumentos, preços, disponibilidade, pedidos, entregas e políticas "
            "da loja. Pode reformular a dúvida?"
        )

    def _catalog_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        category_id = self.store.category_id_for(message)
        max_price = parse_budget(message)
        product = self.store.best_product_match(message)
        words = keyword_set(message)
        query_norm = normalize(message)

        brand_products = self.store.brand_products(
            message, category_id=category_id, max_price=max_price
        )
        # Brand/category questions list every matching model, not just best_product_match().
        if brand_products and not query_model_numbers(message):
            brand = normalize(brand_products[0].name).split()[0].title()
            label = self.store.category_name(category_id).lower() if category_id is not None else "produtos"
            lines = [f"Encontrei {len(brand_products)} {label} da {brand} disponíveis:"]
            for item in brand_products:
                promotion = self.store.active_promotion(item)
                price = format_brl(self.store.effective_price(item))
                suffix = f" ({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})" if promotion else ""
                lines.append(f"- {item.name}: {price}{suffix}; {item.stock_quantity} unidade(s) em estoque.")
            return "\n".join(lines), self.store.search_grounding(
                message, category_id=category_id, max_price=max_price, limit=len(brand_products)
            )

        if words & {"promocao", "promocoes", "desconto", "descontos", "oferta", "ofertas", "especial", "especiais", "exclusivo", "exclusiva", "black", "friday"} and product is None and category_id is None and max_price is None:
            promotions = self.store.active_promotions()
            if not promotions:
                return "No momento, não encontrei promoções ativas em produtos disponíveis.", []
            lines = [
                "Não encontrei uma campanha chamada Black Friday cadastrada, mas estas promoções estão ativas em estoque:"
                if words & {"black", "friday"}
                else "Estas são algumas promoções ativas em estoque:"
            ]
            for item, promotion in promotions:
                lines.append(
                    f"- {item.name}: {format_brl(self.store.effective_price(item))} "
                    f"({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})."
                )
            grounding = [
                RetrievedChunk(
                    title=item.name,
                    source_type="catalog_promotion",
                    content=(
                        f"Produto: {item.name}\nPreço atual: {format_brl(self.store.effective_price(item))}\n"
                        f"Preço original: {format_brl(item.price_brl)}\nDesconto: {promotion.discount_percent:g}%\n"
                        f"Promoção: {promotion.description}\nEstoque: {item.stock_quantity}"
                    ),
                    score=12.0,
                    retrieval="structured",
                )
                for item, promotion in promotions
            ]
            return "\n".join(lines), grounding

        if product is not None and (
            normalize(product.name) in query_norm
            or len(keyword_set(message) & keyword_set(product.name)) >= 2
        ):
            promotion = self.store.active_promotion(product)
            if not product.available:
                state = "descontinuado" if product.status == "discontinued" else "temporariamente indisponível"
                alternatives = self.store.similar_products(product)
                suffix = (
                    " Alternativas em estoque: "
                    + "; ".join(item.name for item in alternatives)
                    + "."
                    if alternatives else ""
                )
                return (
                    f"O {product.name} está {state}.{suffix}",
                    self.store.search_grounding(product.name, include_unavailable=True),
                )
            price = self.store.effective_price(product)
            price_text = format_brl(price)
            if promotion:
                price_text += f" ({promotion.discount_percent:g}% off; preço original {format_brl(product.price_brl)})"
            elif "pix" in words:
                price_text += f"; no PIX, {format_brl(self.store.pix_price(product) or price)}"
            return (
                f"{product.name}: {price_text}. Temos {product.stock_quantity} unidade(s) em estoque.",
                self.store.search_grounding(product.name),
            )

        products = self.store.search_products(
            message, category_id=category_id, max_price=max_price, limit=5
        )
        if not products:
            if category_id is not None and max_price is not None:
                return (
                    f"Não encontrei {self.store.category_name(category_id).lower()} disponíveis até {format_brl(max_price)}.",
                    [],
                )
            return "Não encontrei produtos disponíveis com esses critérios.", []

        label = self.store.category_name(category_id).lower() if category_id is not None else "produtos"
        budget = f" até {format_brl(max_price)}" if max_price is not None else ""
        lines = [f"Encontrei {label} disponíveis{budget}:"]
        for item in products:
            promotion = self.store.active_promotion(item)
            price_text = format_brl(self.store.effective_price(item))
            if promotion:
                price_text += f" ({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})"
            lines.append(f"- {item.name}: {price_text}; {item.stock_quantity} em estoque.")
        return "\n".join(lines), self.store.search_grounding(
            message, category_id=category_id, max_price=max_price
        )

    def _order_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        reference = extract_order_reference(message)
        if self.customer_id is None:
            return (
                "Para proteger seus dados, informe o número do pedido e o nome ou e-mail usado na compra.",
                [],
            )
        order = self.store.order_by_reference(reference) if reference else self.store.latest_order(self.customer_id)
        if order is None:
            return "Não encontrei esse pedido no cadastro. Confira o número informado.", []
        if order.customer_id != self.customer_id:
            return (
                "Não encontrei esse pedido vinculado ao seu cadastro. Confira o número ou o e-mail usado na compra.",
                [],
            )

        answer = f"O pedido {order.order_id} está {STATUS_LABELS.get(order.status, order.status)}."
        if order.estimated_delivery:
            answer += f" Previsão de entrega: {order.estimated_delivery}."
        if order.tracking_code:
            answer += f" Rastreamento: {order.tracking_code}."
        elif order.status in {"pending", "confirmed"}:
            answer += " O código de rastreamento será enviado quando o pedido for despachado."
        items = self.store.order_items_for(order.order_id)
        if items:
            answer += " Itens: " + ", ".join(f"{qty}x {name}" for _, name, qty in items) + "."
        return answer, [
            RetrievedChunk(
                title=f"Pedido {order.order_id}",
                source_type="order_record",
                content=(
                    f"Pedido: {order.order_id}\nStatus: {STATUS_LABELS.get(order.status, order.status)}\n"
                    f"Previsão: {order.estimated_delivery or 'não informada'}\n"
                    f"Rastreamento: {order.tracking_code or 'ainda não despachado'}\n"
                    f"Itens: {', '.join(f'{qty}x {name}' for _, name, qty in items)}"
                ),
                score=10.0,
                retrieval="keyword",
            )
        ]

    def _policy_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.rag.search(message)
        words = keyword_set(message)
        if words & {"horario", "horarios", "expediente", "funcionamento", "aberto"}:
            answer = "O atendimento funciona de segunda a sexta, das 9h às 18h; sábado, das 9h às 13h; domingo e feriados: fechado."
        elif words & {"endereco", "localizacao", "onde"}:
            answer = "A loja fica na Rua 14 de Maio, 3200, Centro, Campo Grande/MS, CEP 79202-333."
        elif words & {"pagamento", "pix", "boleto", "cartao", "parcelamento"}:
            answer = "Aceitamos PIX, débito, crédito em até 12x sem juros e boleto. O PIX tem 5% de desconto, mas não é cumulativo com promoções."
        elif words & {"devolucao", "devolver", "arrependimento", "reembolso"}:
            answer = "Em compras online, você pode pedir devolução em até 7 dias corridos após o recebimento, sem justificativa. O produto deve estar sem uso e na embalagem original; o reembolso ocorre na forma de pagamento original em até 10 dias úteis."
        elif words & {"troca", "trocar", "preferencia", "modelo", "cor"}:
            answer = "Trocas por preferência são permitidas em até 7 dias, conforme disponibilidade, com o produto em perfeito estado e na embalagem original."
        elif words & {"defeito", "garantia"}:
            answer = "Defeitos de fabricação podem ser tratados em até 30 dias para troca; todos os produtos também têm garantia legal de 90 dias a partir do recebimento. Mau uso, quedas e modificações não autorizadas não são cobertos."
        elif words & {"frete", "entrega", "entregas"}:
            answer = "Na região metropolitana de Campo Grande, o frete é grátis acima de R$ 500 e custa R$ 35 abaixo disso, com prazo de 1 a 3 dias úteis. Para outras cidades, o valor depende do CEP, peso e dimensões."
        elif words & {"privacidade", "lgpd", "dados"}:
            answer = "Os dados são usados para processar pedidos, comunicar status e cumprir obrigações legais; não são compartilhados para marketing sem consentimento. A exclusão pode ser solicitada pelo WhatsApp ou e-mail."
        elif words & {"reclamacao", "reclamar", "problema"}:
            answer = "Sinto muito pelo transtorno. Vou registrar a reclamação e encaminhar para a equipe responsável; o prazo de retorno é de até 24 horas úteis."
        elif chunks:
            excerpt = re.sub(r"\s+", " ", chunks[0].content)
            answer = f"Encontrei esta orientação no manual: {excerpt[:500].rstrip()}..."
        else:
            answer = "Não encontrei uma regra específica na política disponível. A equipe precisa confirmar essa orientação."
        if chunks:
            answer += f" (Manual de Políticas, p. {chunks[0].page}.)"
        return answer, chunks


def extract_order_reference(message: str) -> Optional[str]:
    tracking = re.search(r"\bBR[A-Z0-9]{5,}[A-Z0-9]BR\b", message.upper())
    if tracking:
        return tracking.group(0)
    match = re.search(
        r"(?:pedido|ordem|compra|#)\s*(?:n[ºo.]?\s*)?(\d+)\b",
        message,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def is_greeting(message: str) -> bool:
    return bool(re.fullmatch(r"\s*(oi|ola|olá|bom dia|boa tarde|boa noite|hello|hi)[!. ]*", message, re.I))


def is_injection(message: str) -> bool:
    normalized = normalize(message)
    return bool(
        re.search(
            r"ignore (as )?instrucoes|ignore previous|revele|mostre.*prompt|system prompt|"
            r"voce agora e|you are now|jailbreak|developer mode",
            normalized,
        )
    )


def route_message(message: str, store: CatalogStore) -> str:
    if not normalize(message):
        return "unknown"
    if is_greeting(message):
        return "greeting"
    if is_injection(message):
        return "out_of_scope"

    words = keyword_set(message)
    # Promotion language is catalog intent, never policy intent.
    if words & {"promocao", "promocoes", "desconto", "descontos", "oferta", "ofertas", "especial", "especiais", "exclusivo", "exclusiva", "black", "friday"}:
        return "catalog"

    accessory_only = bool(words & ACCESSORY_TERMS) and not bool(
        words & {
            "instrumento", "violao", "violoes", "guitarra", "guitarras", "baixo",
            "teclado", "piano", "bateria", "ukulele",
        }
    )
    if accessory_only:
        return "out_of_scope"

    order_signal = bool(
        words & {"pedido", "pedidos", "ordem", "compra", "rastrear", "rastreamento", "tracking", "status", "despachado"}
    )
    policy_signal = bool(
        words & {
            "devolucao", "devolver", "arrependimento", "troca", "trocar", "defeito",
            "garantia", "pagamento", "pix", "boleto", "cartao", "horario",
            "expediente", "endereco", "localizacao", "frete", "entrega",
            "privacidade", "lgpd", "reembolso", "reclamacao", "problema",
        }
    )
    if order_signal and (
        words & {"status", "rastrear", "rastreamento", "despachado"}
        or extract_order_reference(message) is not None
    ):
        return "order"
    if policy_signal:
        return "policy"

    catalog_signal = bool(
        words & {
            "preco", "custa", "valor", "estoque", "disponivel", "disponibilidade",
            "opcoes", "produto", "catalogo", "promocao", "promocoes", "desconto",
            "descontos", "oferta", "ofertas", "especial", "especiais", "exclusivo", "exclusiva", "black", "friday", "tem",
        }
    )
    if store.category_id_for(message) is not None or store.best_product_match(message) is not None or catalog_signal:
        return "catalog"
    if order_signal:
        return "order"
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente de atendimento da Empório da Música")
    default_dir = Path(__file__).resolve().parent / "data"
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    parser.add_argument("--customer-id", type=int)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--web", action="store_true", help="inicia a interface web local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--message", action="append")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.web:
            from web import serve
            serve(args.data_dir, host=args.host, port=args.port, use_llm=not args.no_llm, model=args.model)
            return 0
        agent = StoreAgent(
            args.data_dir,
            customer_id=args.customer_id,
            use_llm=not args.no_llm,
            model=args.model,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Erro ao carregar dados: {exc}", file=sys.stderr)
        return 1

    if args.message:
        for message in args.message:
            print(agent.handle(message))
        return 0

    print(f"Empório da Música - Gemini {agent.client.model}")
    print("Digite 'sair' para encerrar.\n")
    while True:
        try:
            message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if normalize(message) in {"sair", "exit", "quit"}:
            break
        print(f"Agente: {agent.handle(message)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








