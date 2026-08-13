"""Structured catalog access and shared retrieval data models.

The catalog is the source of truth for product, promotion, customer, and order
facts. It is deliberately separate from policy RAG so exact prices and stock
values are never inferred from semantic similarity.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
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
    "deveolucao": {"devolucao", "reembolso"},
    # Customers conjugate this verb freely: "vocês parcelam?", "dá pra parcelar?",
    # "em quantas parcelas?". All of them are the payment policy question.
    "parcelado": {"parcelamento", "pagamento", "cartao"},
    "parcelar": {"parcelamento", "pagamento", "cartao"},
    "parcelam": {"parcelamento", "pagamento", "cartao"},
    "parcelamos": {"parcelamento", "pagamento", "cartao"},
    "parcela": {"parcelamento", "pagamento", "cartao"},
    "parcelas": {"parcelamento", "pagamento", "cartao"},
    "juros": {"parcelamento", "pagamento"},
    "pagar": {"pagamento"},
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
    normalized = normalize(message).replace("oate", "ate")
    match = re.search(
        r"(?:ate|menos de|abaixo de|no maximo|por ate)"
        r"\s*(?:r\$)?\s*([\d.]+(?:,[\d]{1,2})?)",
        normalized,
        re.IGNORECASE,
    )
    return parse_number(match.group(1)) if match else None


def strip_budget_constraints(message: str) -> str:
    normalized = normalize(message).replace("oate", "ate")
    return " ".join(
        re.sub(
            r"(?:ate|menos de|abaixo de|no maximo|por ate)"
            r"\s*(?:r\$)?\s*[\d.]+(?:,[\d]{1,2})?\s*(?:reais?|rs)?",
            " ",
            normalized,
            flags=re.IGNORECASE,
        ).split()
    )


def merge_catalog_context(
    previous_query: str,
    current_message: str,
    *,
    replace_budget: bool = False,
) -> str:
    base = (
        strip_budget_constraints(previous_query)
        if replace_budget
        else normalize(previous_query)
    )
    return " ".join(part for part in (base, normalize(current_message)) if part).strip()

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

        # Short, common typing errors such as "violã" -> "viola" should
        # still resolve to the structured category, but only against known
        # category aliases.
        tokens = re.findall(r"[a-z0-9]+", normalized)
        close_alias = next(
            (
                candidate
                for token in tokens
                for candidate in difflib.get_close_matches(
                    token, list(CATEGORY_ALIASES), n=1, cutoff=0.82
                )
            ),
            None,
        )
        if close_alias is not None:
            category = CATEGORY_ALIASES[close_alias]
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



__all__ = [
    "CATEGORY_ALIASES",
    "CatalogStore",
    "Customer",
    "Order",
    "Product",
    "Promotion",
    "QUERY_SYNONYMS",
    "RetrievedChunk",
    "STATUS_LABELS",
    "STOPWORDS",
    "format_brl",
    "keyword_set",
    "keyword_tokens",
    "locate_csv",
    "merge_catalog_context",
    "normalize",
    "parse_budget",
    "parse_int",
    "parse_number",
    "parse_specs",
    "query_model_numbers",
    "read_csv",
    "strip_budget_constraints",
]