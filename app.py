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
    "yahama": {"yamaha"},
    "descomto": {"desconto"},
    "descontoo": {"desconto"},
    "promocaoo": {"promocao", "promocoes"},
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

# Situations the agent cannot settle on its own, which belong to a human agent.
# Deliberately narrow: a generic "reclamação" stays a policy answer; only a late
# delivery, damaged goods or an explicit request for a person escalates.
HANDOFF_PATTERNS = (
    r"\batras(?:ad[oa]s?|o|os|ou|ando)\b",
    r"\bnao (?:chegou|chegaram|recebi|veio|vieram|entregaram|foi entregue)\b",
    r"\b(?:falar|conversar|atendimento) com (?:um |uma )?(?:atendente|humano|humana|pessoa|gerente|vendedor|alguem)\b",
    r"\b(?:quero|queria|preciso de|passa para) (?:um |uma )?(?:atendente|humano|humana|gerente)\b",
    r"\b(?:veio|chegou|recebi) (?:o |a )?(?:produto |item |pedido )?(?:quebrad[oa]|danificad[oa]|errad[oa]|amassad[oa])\b",
    r"\bprocon\b",
)

# Buying is a different flow from an order lookup: it creates no order and loads
# no customers/order_items. It keeps the chosen product and collects hand-off data.
CHECKOUT_PATTERNS = (
    r"\b(?:finalizar|concluir|fechar|prosseguir)\b.*\b(?:compra|pedido)\b",
    r"\b(?:quero|gostaria de|posso)\s+(?:comprar|adquirir|levar)\b",
)

# Readable version of HANDOFF_PATTERNS, rendered by the web UI. Keep both lists
# aligned: this is what the customer reads to know what reaches a human.
HANDOFF_TRIGGER_LABELS = (
    ("Pedido atrasado", "meu violão está atrasado"),
    ("Pedido não entregue", "o pedido não chegou"),
    ("Produto com avaria", "chegou quebrado"),
    ("Pedido explícito de atendente", "quero falar com um atendente"),
    ("Menção a órgão de defesa do consumidor", "vou acionar o Procon"),
)


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
    normalized = normalize(value)
    # A budget ("até R$ 1000") is a price, not a model number. Left in, it makes
    # the query look model-specific and suppresses brand listings.
    normalized = re.sub(
        r"(?:ate|menos de|abaixo de|no maximo|por ate)\s*(?:r\$)?\s*[\d.,]+", " ", normalized
    )
    normalized = re.sub(r"r\$\s*[\d.,]+", " ", normalized)
    result: list[str] = []
    for token in re.findall(r"[a-z0-9]+", normalized):
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
        for token in query_words - brand_terms:
            if len(token) >= 4:
                close = difflib.get_close_matches(token, brand_terms, n=1, cutoff=0.78)
                if close:
                    requested_brands.add(close[0])
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

    def grounding_for_products(self, products: Iterable[Product]) -> list[RetrievedChunk]:
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
                details.append(f"Promoção ativa: {promotion.discount_percent:g}% ({promotion.description})")
            chunks.append(
                RetrievedChunk(
                    title=product.name,
                    source_type="catalog_row",
                    content="\n".join(details),
                    score=10.0,
                    retrieval="structured",
                )
            )
        return chunks

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
        return self.grounding_for_products(products)


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
        handoff_enabled: bool = True,
    ):
        data_path = Path(data_dir)
        self.store = CatalogStore(data_path)
        self.client = GeminiClient(api_key=None if use_llm else "", model=model or GEMINI_MODEL)
        self.rag = PolicyRAG(data_path / "políticas_da_loja.pdf", self.client)
        self.customer_id = customer_id
        self.history: list[dict[str, str]] = []
        self.use_llm = use_llm
        # Toggled by the customer in the web UI; when off, the agent answers
        # normally instead of escalating.
        self.handoff_enabled = handoff_enabled
        self.pending_handoffs: list[dict[str, Any]] = []
        self.last_catalog_products: list[Product] = []
        self.last_catalog_offset = 0
        self.last_catalog_label = "produtos"
        self.last_catalog_query = ""
        self.last_selected_product: Optional[Product] = None
        self.checkout_active = False
        self.checkout_product_id: Optional[int] = None
        self.checkout_address: Optional[str] = None
        self.checkout_quantity = 1
        self.checkout_confirmed = False
        self.exchange_active = False
        self.exchange_product_id: Optional[int] = None
        self.exchange_received_date: Optional[str] = None

    @property
    def customer(self) -> Optional[Customer]:
        return self.store.customers_by_id.get(self.customer_id) if self.customer_id else None

    def _remember_customer(self, message: str) -> None:
        # Identity lookup is opt-in; catalog/promotion/checkout questions do not load customers.
        if self.customer_id is not None or self.checkout_active:
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
        catalog_follow_up = is_catalog_follow_up(message)
        catalog_message = message
        previous_user = self.last_catalog_query or next(
            (item["content"] for item in reversed(self.history) if item["role"] == "user"),
            "",
        )
        if catalog_follow_up and previous_user and (self.last_catalog_query or route_message(previous_user, self.store) == "catalog"):
            catalog_message = previous_user + " " + message
        exchange_continuation = self.exchange_active and is_exchange_continuation(message, self.store)
        checkout_continuation = self.checkout_active and is_checkout_continuation(message, self.store)
        intent = (
            "exchange"
            if exchange_continuation
            else "checkout"
            if checkout_continuation
            else route_message(catalog_message, self.store, allow_handoff=self.handoff_enabled)
        )

        if intent == "greeting":
            answer = self._greeting()
            grounding: list[RetrievedChunk] = []
        elif intent == "out_of_scope":
            answer = self._out_of_scope()
            grounding = []
        elif intent == "injection":
            answer = self._injection()
            grounding = []
        elif intent == "handoff":
            answer, grounding = self._handoff_answer(message)
        elif intent == "checkout":
            answer, grounding = self._checkout_answer(message)
        elif intent == "exchange":
            answer, grounding = self._exchange_answer(message)
        elif intent == "catalog":
            answer, grounding = self._catalog_answer(catalog_message)
            brand_switch_follow_up = bool(re.fullmatch(r"e(?: da| do| das| dos| a| o)? [a-z0-9-]+", normalize(message).strip("?!., ")))
            if not catalog_follow_up or brand_switch_follow_up:
                self.last_catalog_query = catalog_message
        elif intent == "order":
            answer, grounding = self._order_answer(message)
        elif intent == "policy":
            answer, grounding = self._policy_answer(message)
        else:
            answer, grounding = self._unknown(), []

        # Only rewrite when there is retrieved evidence to rewrite from. With no
        # grounding the answer is a deterministic refusal ("identify yourself",
        # "no rule found"), and a free rewrite drops the guarantee it carries.
        if (self.use_llm and self.client.enabled and grounding and intent in {"catalog", "order", "policy", "unknown"} and not self._is_structured_catalog_list(intent, grounding)):
            generated = self._gemini_answer(catalog_message if intent == "catalog" else message, grounding)
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
            if re.search(r"\b(?:quais|todos|todas|cada|lista completa|mais modelos|so tem esses|so esses|tem mais|tem outros|sao todos|sao esses)\b", normalize(message))
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

    @staticmethod
    def _is_structured_catalog_list(
        intent: str, grounding: list[RetrievedChunk]
    ) -> bool:
        """Keep exact multi-row catalog answers out of free-form summarization."""
        return bool(
            intent == "catalog"
            and len(grounding) > 1
            and all(
                chunk.source_type in {"catalog_row", "catalog_promotion"}
                for chunk in grounding
            )
        )

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
    def _injection() -> str:
        # Holds scope without repeating the accessory message, which reads as a
        # non-sequitur here, and without naming what is being protected.
        return (
            "Não consigo compartilhar minhas configurações de atendimento. Posso ajudar "
            "com instrumentos musicais, preços, estoque, pedidos e políticas da loja."
        )

    @staticmethod
    def _unknown() -> str:
        return (
            "Posso ajudar com instrumentos, preços, disponibilidade, pedidos, entregas e políticas "
            "da loja. Pode reformular a dúvida?"
        )

    def _checkout_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        """Keep checkout conversational without pretending an order was created.

        Catalog facts come from the structured product index. Payment and delivery
        guidance comes from local PDF RAG; customer/order tables are not needed.
        """
        self.checkout_active = True
        policy_chunks = self.rag.search(
            "finalizar compra formas de pagamento entrega endereço"
        )
        # Naming a product now counts as choosing it; otherwise fall back to what
        # was already chosen, then to the single product the customer just viewed.
        named = self.store.best_product_match(message)
        product = named
        if named is not None and named.available:
            self.checkout_product_id = named.product_id
        elif named is None:
            if self.checkout_product_id is not None:
                product = self.store.products_by_id.get(self.checkout_product_id)
            elif self.last_selected_product is not None:
                product = self.last_selected_product
                self.checkout_product_id = product.product_id

        address = extract_delivery_address(message)
        if address:
            self.checkout_address = address
        quantity = extract_quantity(message)
        if quantity is not None:
            self.checkout_quantity = quantity

        grounding = (
            self.store.grounding_for_products([product]) if product is not None else []
        )
        grounding.extend(policy_chunks)

        if product is None:
            return (
                "Claro! Para iniciar a compra, me diga qual modelo você quer adquirir. "
                "Como a lista anterior tinha várias opções, não vou escolher automaticamente "
                "o primeiro item. Você pode enviar o nome ou modelo, por exemplo: Yamaha C40.",
                grounding,
            )
        if not product.available:
            return (
                f"O {product.name} está temporariamente indisponível e não pode ser incluído "
                "na compra agora. Posso procurar uma alternativa em estoque.",
                grounding,
            )

        price = format_brl(self.store.effective_price(product))
        item = f"{self.checkout_quantity}x {product.name} ({price})"
        if address:
            return (
                f"Recebi o endereço de entrega: {self.checkout_address}. "
                f"Para encaminhar a compra de {item}, faltam seu nome completo, telefone ou e-mail "
                "e a forma de pagamento (PIX, débito, cartão ou boleto). O pedido ainda não foi criado; "
                "ele só será registrado após essa confirmação.",
                grounding,
            )
        # Chosen explicitly (named or confirmed) versus merely inferred from the
        # previous turn, which still has to be checked with the customer.
        if named is not None or is_checkout_confirmation(message) or self.checkout_confirmed:
            self.checkout_confirmed = True
            shortage = (
                f" Temos {product.stock_quantity} unidade(s) em estoque, então posso seguir "
                f"com {product.stock_quantity} ou buscar outro modelo."
                if self.checkout_quantity > product.stock_quantity
                else f" Temos {product.stock_quantity} unidade(s) disponíveis."
            )
            return (
                f"Perfeito! Selecionei {item}.{shortage} "
                "Para concluir, envie seu nome completo, telefone ou e-mail "
                "e o endereço de entrega. Depois confirmamos a forma de pagamento e registramos o pedido.",
                grounding,
            )
        return (
            f"Encontrei {item}; temos {product.stock_quantity} unidade(s) disponíveis. "
            "Você quer esse modelo? Se sim, confirme a quantidade e envie nome completo, "
            "telefone ou e-mail e endereço de entrega.",
            grounding,
        )
    def _exchange_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        """Continue an exchange case while keeping facts split by source.

        The product row is deterministic catalog grounding. The exchange rule is
        retrieved from the policy PDF. No customer, order, or order-item table is
        needed to clarify eligibility.
        """
        self.exchange_active = True
        policy_chunks = self.rag.search(
            "troca por preferência 7 dias corridos após recebimento "
            "produto perfeito estado embalagem original disponibilidade"
        )
        named = self.store.best_product_match(message)
        product = named
        if named is not None:
            self.exchange_product_id = named.product_id
        elif self.exchange_product_id is not None:
            product = self.store.products_by_id.get(self.exchange_product_id)

        received_date = extract_exchange_date(message)
        if received_date:
            self.exchange_received_date = received_date

        grounding = (
            self.store.grounding_for_products([product]) if product is not None else []
        )
        grounding.extend(policy_chunks)

        if product is None:
            return (
                "Entendi. Para verificar a troca por preferência, informe o modelo do "
                "violão e a data em que você recebeu o pedido. A política prevê até "
                "7 dias corridos após o recebimento, com o produto em perfeito estado "
                "e na embalagem original.",
                grounding,
            )

        price = format_brl(self.store.effective_price(product))
        stock = (
            f" O catálogo registra {product.stock_quantity} unidade(s) disponíveis."
            if product.available
            else " O catálogo registra que esse modelo está indisponível no momento."
        )
        if self.exchange_received_date:
            return (
                f"Encontrei o {product.name}: {price}.{stock} Para troca por preferência, "
                "a política permite solicitar em até 7 dias corridos após o recebimento, "
                "conforme disponibilidade, com o instrumento em perfeito estado e na "
                f"embalagem original. Você informou {self.exchange_received_date}; essa "
                "data precisa ser a de recebimento, e ainda preciso confirmar o ano. "
                "Também me diga qual modelo deseja no lugar; eventual diferença de valor "
                "pode ser cobrada ou reembolsada.",
                grounding,
            )
        return (
            f"Encontrei o {product.name}: {price}.{stock} Para validar a troca por "
            "preferência, preciso da data de recebimento, do estado do instrumento e "
            "do modelo que você deseja no lugar. A política exige até 7 dias corridos "
            "após o recebimento e a embalagem original.",
            grounding,
        )
    def _handoff_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        """Build the transfer request for a human agent.

        NOT FINISHED: the real dispatch (support queue, CRM, the store's
        WhatsApp) does not exist yet. For now the ticket is only collected in
        self.pending_handoffs and the customer gets the confirmation in text.
        """
        customer = self.customer
        order = self.store.latest_order(self.customer_id) if self.customer_id else None
        ticket = {
            "protocol": f"H{len(self.pending_handoffs) + 1:04d}",
            "message": message,
            "customer_id": self.customer_id,
            "customer_name": customer.name if customer else None,
            "order_id": order.order_id if order else None,
            "history": list(self.history[-6:]),
        }
        self.pending_handoffs.append(ticket)
        # TODO: dispatch the ticket to the human queue and return the real protocol.

        answer = (
            f"Sinto muito pelo transtorno. Já estou acionando um atendente da equipe "
            f"para assumir o seu caso (protocolo {ticket['protocol']})."
        )
        if order is not None:
            answer += f" Vou encaminhar junto os dados do pedido {order.order_id}."
        elif self.customer_id is None:
            answer += (
                " Para agilizar, me informe o número do pedido e o nome ou e-mail "
                "usado na compra."
            )
        answer += " O retorno acontece em até 24 horas úteis, no horário de atendimento."
        return answer, []

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
        if brand_products and product is None and not query_model_numbers(message):
            self.last_selected_product = None
            brand = normalize(brand_products[0].name).split()[0].title()
            label = self.store.category_name(category_id).lower() if category_id is not None else "produtos"
            self.last_catalog_products = brand_products
            self.last_catalog_offset = len(brand_products)
            self.last_catalog_label = label
            lines = [f"Encontrei {len(brand_products)} {label} da {brand} disponíveis:"]
            for item in brand_products:
                promotion = self.store.active_promotion(item)
                price = format_brl(self.store.effective_price(item))
                suffix = f" ({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})" if promotion else ""
                lines.append(f"- {item.name}: {price}{suffix}; {item.stock_quantity} unidade(s) em estoque.")
            return "\n".join(lines), self.store.grounding_for_products(brand_products)

        if words & {"promocao", "promocoes", "desconto", "descontos", "oferta", "ofertas", "especial", "especiais", "exclusivo", "exclusiva", "black", "friday"} and product is None and category_id is None and max_price is None:
            promotions = self.store.active_promotions()
            if not promotions:
                return "No momento, não encontrei promoções ativas em produtos disponíveis.", []
            lines = [
                "A política prevê Black Friday em novembro, com descontos de 15% a 30%. No catálogo atual não há uma promoção ativa identificada como Black Friday. Estas promoções estão ativas em estoque:"
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
            self.last_selected_product = product
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

        all_products = self.store.search_products(
            message, category_id=category_id, max_price=max_price,
            limit=max(100, len(self.store.products)),
        )
        if not all_products:
            if category_id is not None and max_price is not None:
                return (
                    f"Não encontrei {self.store.category_name(category_id).lower()} disponíveis até {format_brl(max_price)}.",
                    [],
                )
            return "Não encontrei produtos disponíveis com esses critérios.", []

        label = self.store.category_name(category_id).lower() if category_id is not None else "produtos"
        budget = f" até {format_brl(max_price)}" if max_price is not None else ""
        normalized_query = normalize(message)
        follow_up = is_catalog_follow_up(message)
        previous_ids = {item.product_id for item in self.last_catalog_products}
        current_ids = {item.product_id for item in all_products}
        same_context = bool(previous_ids) and previous_ids == current_ids

        if category_id is not None and follow_up and same_context:
            if re.search(r"\b(?:so tem|so esses|sao esses|apenas|somente)\b", normalized_query):
                shown = min(self.last_catalog_offset or 5, len(all_products))
                return (
                    f"Não. Encontrei {len(all_products)} {label} disponíveis{budget}. "
                    f"A lista anterior mostrava apenas {shown} opções; posso filtrar por marca, tipo ou faixa de preço.",
                    self.store.grounding_for_products(all_products[:shown]),
                )
            if re.search(r"\b(?:tem mais|mais|outros)\b", normalized_query):
                start_offset = min(self.last_catalog_offset, len(all_products))
                page = all_products[start_offset:start_offset + 5]
                if page:
                    self.last_catalog_offset = start_offset + len(page)
                    lines = [f"Sim, há mais {label} disponíveis:"]
                    for item in page:
                        promotion = self.store.active_promotion(item)
                        price_text = format_brl(self.store.effective_price(item))
                        if promotion:
                            price_text += f" ({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})"
                        lines.append(f"- {item.name}: {price_text}; {item.stock_quantity} em estoque.")
                    return "\n".join(lines), self.store.grounding_for_products(page)
                return f"Já mostrei todas as {label} disponíveis nesta busca ({len(all_products)} no total).", self.store.grounding_for_products(all_products)

        if category_id is not None:
            self.last_selected_product = None
            products = all_products[:5]
            self.last_catalog_products = all_products
            self.last_catalog_offset = len(products)
            self.last_catalog_label = label
            lines = [f"Encontrei {len(all_products)} {label} disponíveis{budget}."]
            if len(all_products) > len(products):
                lines.append(f"Mostrando {len(products)} opções; posso listar mais ou filtrar por marca, tipo ou orçamento.")
        else:
            products = all_products[:5]
            lines = [f"Encontrei {len(all_products)} {label} disponíveis{budget}:"]
        for item in products:
            promotion = self.store.active_promotion(item)
            price_text = format_brl(self.store.effective_price(item))
            if promotion:
                price_text += f" ({promotion.discount_percent:g}% off; era {format_brl(item.price_brl)})"
            lines.append(f"- {item.name}: {price_text}; {item.stock_quantity} em estoque.")
        return "\n".join(lines), self.store.grounding_for_products(products)

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
            answer = "Aceitamos PIX (pagamento à vista com 5% de desconto sobre o preço de tabela, mas não é cumulativo), débito, crédito em até 12x sem juros (parcela mínima de R$ 100,00) e boleto à vista. De 4x a 6x, a parcela mínima é R$ 80,00; de 7x a 12x, R$ 100,00. É permitida a combinação de formas de pagamento, como PIX + cartão, para compras acima de R$ 2.000,00."
        elif is_exchange_request(message):
            self.exchange_active = True
            self.exchange_product_id = None
            self.exchange_received_date = None
            answer = (
                "Entendi. Para troca por preferência, a política permite solicitar em "
                "até 7 dias corridos após o recebimento, conforme disponibilidade, com "
                "o instrumento em perfeito estado e na embalagem original. Informe o "
                "modelo, a data de recebimento e qual modelo você deseja no lugar."
            )
        elif words & {"devolucao", "devolver", "arrependimento", "reembolso"}:
            answer = "Em compras online, você pode pedir devolução em até 7 dias corridos após o recebimento, sem justificativa. O produto deve estar na embalagem original, sem sinais de uso e com todos os acessórios e manuais. O reembolso ocorre na forma de pagamento original em até 10 dias úteis; o frete de devolução é por conta da loja em caso de arrependimento."
        elif words & {"troca", "trocar", "preferencia", "modelo", "cor"}:
            answer = "Trocas por preferência são permitidas em até 7 dias, conforme disponibilidade, com o produto em perfeito estado e na embalagem original."
        elif words & {"defeito", "garantia"}:
            answer = "Defeitos de fabricação podem ser tratados em até 30 dias corridos para troca. Após esse prazo, o cliente deve acionar a garantia diretamente com o fabricante; a loja pode intermediar. A garantia legal é de 90 dias, e a garantia do fabricante pode variar de 6 meses a 2 anos. Mau uso, quedas, umidade extrema e modificações não autorizadas não são cobertos."
        elif words & {"frete", "entrega", "entregas"}:
            answer = "Na região metropolitana de Campo Grande, o frete é grátis acima de R$ 500 e custa R$ 35 abaixo disso, com prazo de 1 a 3 dias úteis; a entrega é feita por motoboy próprio; o cliente é contactado por telefone antes do despacho. Para outras cidades, o cálculo depende de CEP, peso e dimensões: PAC leva 5 a 12 dias úteis, SEDEX 2 a 5 e Jadlog 3 a 8, todos com seguro contra extravios."
        elif words & {"privacidade", "lgpd", "dados", "exclusao"}:
            answer = "Os dados são usados para processar pedidos, comunicar status e cumprir obrigações legais; não são compartilhados para marketing sem consentimento. A exclusão pode ser solicitada pelo WhatsApp ou e-mail."
        elif words & {"black", "friday"}:
            answer = "A política prevê a campanha Black Friday em novembro, com descontos de 15% a 30% no catálogo. As condições precisam ser confirmadas nas promoções ativas, e os descontos não são cumulativos."
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


def is_catalog_follow_up(message: str) -> bool:
    normalized = normalize(message).strip("?!., ")
    return bool(
        re.search(r"\b(?:todos|todas|cada|lista completa|mais modelos?|so tem esses|so esses|tem mais|tem outros|sao todos|sao esses|apenas esses|somente esses)\b", normalized)
        or re.fullmatch(r"e(?: da| do| das| dos| a| o)? [a-z0-9-]+", normalized)
    )

def is_checkout_request(message: str) -> bool:
    normalized = normalize(message)
    return any(re.search(pattern, normalized) for pattern in CHECKOUT_PATTERNS)


def is_checkout_confirmation(message: str) -> bool:
    normalized = normalize(message).strip("?!., ")
    return bool(
        re.fullmatch(
            r"(?:sim|sim isso|isso|esse|essa|pode ser|e esse|e essa|é esse|e esse mesmo|confirmo|confirmado)",
            normalized,
        )
    )


def extract_delivery_address(message: str) -> Optional[str]:
    """Recognize a street address without interpreting it as a product query.

    Returns only the address span: the customer usually wraps it in politeness
    or a payment note, and echoing the whole sentence back reads as a mistake.
    """
    match = re.search(
        r"\b(?:rua|avenida|av|alameda|travessa|estrada|rodovia|pra[cç]a)\b.*",
        message,
        re.IGNORECASE,
    )
    if not match or not re.search(r"\d{1,6}", match.group(0)):
        return None
    address = re.split(
        r",?\s+(?:e\s+(?:quero|queria|vou|posso|pago)|por favor|obrigad[oa]|"
        r"pode\s+(?:enviar|entregar|mandar)|pagamento|pago)\b",
        match.group(0),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(address.strip(" ,.;").split()) or None


# Spelled-out counts customers actually use; digits cover the rest.
QUANTITY_WORDS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}


def extract_quantity(message: str) -> Optional[int]:
    """Read a checkout quantity, only with an explicit unit or a buying verb.

    Kept strict on purpose: a bare number in "até 1000" is a budget, not a count.
    """
    normalized = normalize(message)
    match = re.search(
        r"\b(\d{1,3})\s*(?:unidade|unidades|peca|pecas|item|itens|x)\b", normalized
    ) or re.search(
        r"\b(?:quero|queria|vou levar|leva|manda|coloca|seriam|sao|quantidade)\s+(\d{1,3})\b",
        normalized,
    )
    if match:
        value = int(match.group(1))
        return value if 1 <= value <= 99 else None
    words = "|".join(QUANTITY_WORDS)
    match = re.search(rf"\b({words})\s+(?:unidade|unidades|peca|pecas|item|itens)\b", normalized)
    return QUANTITY_WORDS[match.group(1)] if match else None


def is_checkout_continuation(message: str, store: CatalogStore) -> bool:
    normalized = normalize(message)
    if extract_delivery_address(message) or is_checkout_confirmation(message):
        return True
    # The agent asks for a quantity, so the answer to it must stay in checkout.
    if extract_quantity(message) is not None:
        return True
    if re.search(r"@|\+?\d[\d ()-]{7,}", message):
        return True
    product = store.best_product_match(message)
    if product is not None and not re.search(
        r"\b(?:quais|qual|preco|custa|valor|estoque|disponivel|tem|opcoes)\b",
        normalized,
    ):
        return True
    return False

def is_exchange_request(message: str) -> bool:
    normalized = normalize(message)
    return bool(
        re.search(r"\bnao\s+(?:gostei|gistei)\b", normalized)
        or re.search(r"\b(?:troca por preferencia|trocar de modelo|mudar de modelo)\b", normalized)
        or re.search(r"\b(?:trocar|troca)\b.{0,80}\b(?:violao|instrumento|modelo|cor)\b", normalized)
        or re.search(r"\b(?:quero|gostaria|preciso|posso|vou)\b.{0,40}\b(?:trocar|fazer uma troca)\b", normalized)
    )


def extract_exchange_date(message: str) -> Optional[str]:
    match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b",
        message,
    )
    if not match:
        return None
    day, month = int(match.group(1)), int(match.group(2))
    if not 1 <= day <= 31 or not 1 <= month <= 12:
        return None
    year = match.group(3)
    if year:
        year = year if len(year) == 4 else f"20{year.zfill(2)}"
        return f"{day:02d}/{month:02d}/{year}"
    return f"{day:02d}/{month:02d}"


def is_exchange_continuation(message: str, store: CatalogStore) -> bool:
    normalized = normalize(message)
    if extract_exchange_date(message):
        return True
    if re.search(
        r"\b(?:recebi|recebimento|comprei|compra|data|quando|embalagem|"
        r"sem uso|perfeito estado|acessorios|manuais)\b",
        normalized,
    ):
        return True
    product = store.best_product_match(message)
    if product is not None and not re.search(
        r"\b(?:quais|qual|preco|custa|valor|estoque|disponivel|tem|opcoes)\b",
        normalized,
    ):
        return True
    return False

def needs_human_handoff(message: str) -> bool:
    normalized = normalize(message)
    return any(re.search(pattern, normalized) for pattern in HANDOFF_PATTERNS)


def is_injection(message: str) -> bool:
    normalized = normalize(message)
    return bool(
        re.search(
            r"ignore (as )?instrucoes|ignore previous|revele|mostre.*prompt|system prompt|"
            r"voce agora e|you are now|jailbreak|developer mode",
            normalized,
        )
    )


def route_message(message: str, store: CatalogStore, *, allow_handoff: bool = True) -> str:
    if not normalize(message):
        return "unknown"
    if is_greeting(message):
        return "greeting"
    if is_injection(message):
        return "injection"
    # Escalation comes before catalog: "meu violão está atrasado" names a
    # product, but it is a support case, not a search.
    if allow_handoff and needs_human_handoff(message):
        return "handoff"

    words = keyword_set(message)
    accessory_only = bool(words & ACCESSORY_TERMS) and not bool(
        words & {
            "instrumento", "violao", "violoes", "guitarra", "guitarras", "baixo",
            "teclado", "piano", "bateria", "ukulele",
        }
    )
    if accessory_only:
        return "out_of_scope"
    if is_checkout_request(message):
        return "checkout"

    order_signal = bool(
        words & {"pedido", "pedidos", "ordem", "compra", "rastrear", "rastreamento", "tracking", "status", "despachado"}
    )
    policy_signal = bool(
        is_exchange_request(message)
        or words & {
            "devolucao", "devolver", "arrependimento", "troca", "trocar", "defeito",
            "garantia", "pagamento", "pix", "boleto", "cartao", "horario",
            "expediente", "endereco", "localizacao", "frete", "entrega",
            "privacidade", "lgpd", "dados", "exclusao", "reembolso", "reclamacao", "problema",
            "politica", "manual", "acumula", "cumulativo",
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









