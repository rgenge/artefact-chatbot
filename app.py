"""Command-line entrypoint for the Empório da Música customer-service agent.

Application logic lives in src/. This file intentionally stays small so the
CLI, HTTP UI, tests, and future integrations all use the same agent path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.agent import (
    ACCESSORY_TERMS,
    CHECKOUT_PATTERNS,
    HANDOFF_PATTERNS,
    HANDOFF_TRIGGER_LABELS,
    StoreAgent,
    extract_checkout_contact,
    extract_checkout_name,
    extract_checkout_payment,
    extract_delivery_address,
    extract_exchange_date,
    extract_order_reference,
    extract_quantity,
    is_catalog_follow_up,
    is_checkout_confirmation,
    is_checkout_continuation,
    is_checkout_request,
    is_contextual_follow_up,
    is_exchange_continuation,
    is_exchange_request,
    is_greeting,
    is_injection,
    needs_human_handoff,
    route_message,
)
from src.catalog import (
    CATEGORY_ALIASES,
    QUERY_SYNONYMS,
    STATUS_LABELS,
    STOPWORDS,
    CatalogStore,
    Customer,
    Order,
    Product,
    Promotion,
    RetrievedChunk,
    format_brl,
    keyword_set,
    keyword_tokens,
    locate_csv,
    merge_catalog_context,
    normalize,
    parse_budget,
    parse_int,
    parse_number,
    parse_specs,
    query_model_numbers,
    read_csv,
    strip_budget_constraints,
)
from src.rag import (
    GEMINI_EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    GeminiClient,
    GeminiError,
    PolicyRAG,
    load_dotenv,
)

__all__ = [
    "ACCESSORY_TERMS",
    "CATEGORY_ALIASES",
    "CHECKOUT_PATTERNS",
    "CatalogStore",
    "Customer",
    "GEMINI_EMBEDDING_DIMENSIONS",
    "GEMINI_EMBEDDING_MODEL",
    "GEMINI_MODEL",
    "GeminiClient",
    "GeminiError",
    "HANDOFF_PATTERNS",
    "HANDOFF_TRIGGER_LABELS",
    "Order",
    "PolicyRAG",
    "Product",
    "Promotion",
    "QUERY_SYNONYMS",
    "RetrievedChunk",
    "STATUS_LABELS",
    "STOPWORDS",
    "StoreAgent",
    "extract_checkout_contact",
    "extract_checkout_name",
    "extract_checkout_payment",
    "extract_delivery_address",
    "extract_exchange_date",
    "extract_order_reference",
    "extract_quantity",
    "format_brl",
    "is_catalog_follow_up",
    "is_checkout_confirmation",
    "is_checkout_continuation",
    "is_checkout_request",
    "is_contextual_follow_up",
    "is_exchange_continuation",
    "is_exchange_request",
    "is_greeting",
    "is_injection",
    "keyword_set",
    "keyword_tokens",
    "load_dotenv",
    "locate_csv",
    "merge_catalog_context",
    "needs_human_handoff",
    "normalize",
    "parse_budget",
    "parse_int",
    "parse_number",
    "parse_specs",
    "query_model_numbers",
    "read_csv",
    "route_message",
    "strip_budget_constraints",
]


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
            from src.web import serve

            serve(
                args.data_dir,
                host=args.host,
                port=args.port,
                use_llm=not args.no_llm,
                model=args.model,
            )
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