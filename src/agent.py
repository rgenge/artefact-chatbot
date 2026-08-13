"""Conversation orchestration for the Empório da Música agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .catalog import (
    STATUS_LABELS,
    CatalogStore,
    Customer,
    Product,
    RetrievedChunk,
    format_brl,
    keyword_set,
    merge_catalog_context,
    normalize,
    parse_budget,
    parse_number,
    query_model_numbers,
)
from .rag import GEMINI_MODEL, GeminiClient, PolicyRAG
ACCESSORY_TERMS = {
    "acessorio", "acessorios", "amplificador", "amplificadores", "cabo",
    "cabos", "case", "cases", "corda", "cordas", "palheta", "palhetas",
    "pedal", "pedais",
}


DAMAGE_PATTERNS = (
    r"\b(?:comprei|recebi|veio|chegou|est[aá])\b.{0,100}\b(?:quebrad[oa]|danificad[oa]|avariad[oa]|rachad[oa]|trincad[oa]|amassad[oa]|solt[oa]|quebrou)\b",
    r"\b(?:perna|parte|peca|suporte|pedal|casco|pele)\b.{0,30}\b(?:quebrad[oa]|danificad[oa]|solt[oa]|rachad[oa]|trincad[oa])\b",
)

HANDOFF_PATTERNS = (
    r"\batras(?:ad[oa]s?|o|os|ou|ando)\b",
    r"\bnao (?:chegou|chegaram|recebi|veio|vieram|entregaram|foi entregue)\b",
    r"\b(?:falar|conversar|atendimento) com (?:um |uma )?(?:atendente|humano|humana|pessoa|gerente|vendedor|alguem)\b",
    r"\b(?:quero|queria|preciso de|passa para) (?:um |uma )?(?:atendente|humano|humana|gerente)\b",
    r"\b(?:veio|chegou|recebi) (?:o |a )?(?:produto |item |pedido )?(?:quebrad[oa]|danificad[oa]|errad[oa]|amassad[oa])\b",
    *DAMAGE_PATTERNS,
    r"\bprocon\b",
)

# Buying is a different flow from an order lookup: it creates no order and loads
# no customers/order_items. It keeps the chosen product and collects hand-off data.
CHECKOUT_PATTERNS = (
    r"\b(?:finalizar|concluir|fechar|prosseguir)\b.*\b(?:compra|pedido)\b",
    r"\b(?:quero|gostaria de|posso)\s+(?:comprar|adquirir|levar)\b",
)

# "What do you sell?" asks for the shape of the catalog, not 61 product rows.
# The honest answer is the category list, which is also short enough to read.
CATEGORY_OVERVIEW_PATTERNS = (
    r"\b(?:que|quais|qual)\s+(?:os\s+|as\s+)?tipos?\b",
    r"\btipos?\s+de\s+(?:produto|produtos|instrumento|instrumentos|coisa|coisas)\b",
    r"\b(?:que|quais)\s+categorias?\b",
    r"\bo\s+que\s+(?:voces|vcs?|a\s+loja)?\s*vend(?:em|e)\b",
    r"\b(?:que|quais)\s+(?:tipo de\s+)?(?:produtos|instrumentos)\s+(?:voces\s+|vcs?\s+)?vend(?:em|e)\b",
    r"\b(?:o\s+que|que)\s+(?:voce|voces|vcs?|a\s+loja)?\s*(?:tem|possui|possuem|oferece|oferecem|vende|vendem)(?:\s+(?:para|pra)\s+vender)?[?!.,\s]*$",
    r"\b(?:o\s+que|que)\s+(?:tem|ha)\s+(?:para|pra)?\s*vender\b",
    r"\btem\s+o\s+que\s+(?:para|pra)?\s*vender\b",
)


def is_category_overview(message: str) -> bool:
    normalized = normalize(message)
    return any(re.search(pattern, normalized) for pattern in CATEGORY_OVERVIEW_PATTERNS)


# Readable version of HANDOFF_PATTERNS, rendered by the web UI. Keep both lists
# aligned: this is what the customer reads to know what reaches a human.
HANDOFF_TRIGGER_LABELS = (
    ("Pedido atrasado", "meu violão está atrasado"),
    ("Pedido não entregue", "o pedido não chegou"),
    ("Produto com avaria", "chegou quebrado"),
    ("Pedido explícito de atendente", "quero falar com um atendente"),
    ("Menção a órgão de defesa do consumidor", "vou acionar o Procon"),
)


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
Fora do catálogo, interprete mensagens curtas usando o histórico e o estado da conversa.
Trate o rascunho seguro como restrição factual: torne-o natural, mas não contradiga
produto selecionado, etapa da compra, política recuperada ou limites operacionais.
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
        self.last_intent: Optional[str] = None
        self.last_policy_query = ""
        self.last_selected_product: Optional[Product] = None
        self.checkout_active = False
        self.checkout_product_id: Optional[int] = None
        self.checkout_address: Optional[str] = None
        self.checkout_quantity = 1
        self.checkout_confirmed = False
        self.checkout_customer_name = ""
        self.checkout_contact = ""
        self.checkout_payment_method = ""
        self.checkout_installments: Optional[int] = None
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
        previous_user = next(
            (item["content"] for item in reversed(self.history) if item["role"] == "user"),
            "",
        )
        previous_intent = self.last_intent or (
            route_message(previous_user, self.store, allow_handoff=self.handoff_enabled)
            if previous_user
            else None
        )
        policy_follow_up = previous_intent == "policy" and is_contextual_follow_up(message)
        rag_query = (
            f"{self.last_policy_query or previous_user} {message}".strip()
            if policy_follow_up
            else message
        )
        # Retrieval precedes routing. It grounds business guidance and helps the
        # LLM understand elliptical turns; catalog facts still come only from CSV.
        policy_chunks = self.rag.search(rag_query)

        catalog_follow_up = is_catalog_follow_up(message)
        catalog_message = message
        catalog_history_available = (
            previous_intent == "catalog" and bool(self.last_catalog_query)
        )
        budget_context_follow_up = (
            catalog_history_available
            and parse_budget(message) is not None
            and self.store.category_id_for(message) is None
        )
        if catalog_history_available and catalog_follow_up:
            catalog_message = merge_catalog_context(self.last_catalog_query, message)
        elif budget_context_follow_up:
            # "e até 600" inherits "violão" from the last catalog turn and
            # replaces, rather than combines with, the old budget.
            catalog_message = merge_catalog_context(
                self.last_catalog_query,
                message,
                replace_budget=True,
            )

        contextual_product = self._recent_catalog_choice(message)
        if self.checkout_active and contextual_product is not None:
            self.last_selected_product = contextual_product
            self.checkout_product_id = contextual_product.product_id
            self.checkout_confirmed = True

        exchange_continuation = self.exchange_active and is_exchange_continuation(message, self.store)
        checkout_continuation = self.checkout_active and (
            contextual_product is not None
            or is_checkout_continuation(message, self.store)
        )
        intent = (
            "exchange"
            if exchange_continuation
            else "checkout"
            if checkout_continuation
            else route_message(catalog_message, self.store, allow_handoff=self.handoff_enabled)
        )
        if (
            intent == "unknown"
            and previous_intent == "policy"
            and is_contextual_follow_up(message)
            and parse_budget(message) is None
        ):
            intent = "policy"

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
            brand_query_follow_up = bool(self.store.brand_products(message))
            if not catalog_follow_up or budget_context_follow_up or brand_switch_follow_up or brand_query_follow_up:
                self.last_catalog_query = catalog_message
        elif intent == "order":
            answer, grounding = self._order_answer(message)
        elif intent == "policy":
            answer, grounding = self._policy_answer(message, policy_chunks)
            if not policy_follow_up:
                self.last_policy_query = message
        else:
            answer, grounding = self._unknown(), policy_chunks

        # The LLM handles language and continuity for non-catalog business flows.
        # Catalog rows, prices and stock are never rewritten by generation.
        llm_eligible = intent in {"policy", "checkout", "exchange", "unknown"} or (
            intent == "order" and bool(grounding)
        )
        if self.use_llm and self.client.enabled and llm_eligible:
            generated = self._gemini_answer(message, grounding, answer, intent)
            if generated and self._response_is_grounded(generated, grounding) and self._response_preserves_draft(generated, answer, self.store):
                answer = generated

        self.last_intent = intent
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _recent_catalog_choice(self, message: str) -> Optional[Product]:
        """Resolve a short choice only against products shown in the last list."""

        if not self.last_catalog_products:
            return None
        normalized = normalize(message)
        if re.search(r"\b(?:quais|qual|tem|preco|custa|estoque|disponivel)\b", normalized):
            return None
        query_words = keyword_set(message) - {
            "ah", "a", "o", "as", "os", "da", "do", "das", "dos", "pode",
            "ser", "esse", "essa", "quero", "queria", "prefiro", "escolho",
        }
        if not query_words:
            return None
        matches = [
            product for product in self.last_catalog_products
            if query_words & keyword_set(product.name)
        ]
        return matches[0] if len(matches) == 1 else None

    def _gemini_answer(
        self,
        message: str,
        grounding: list[RetrievedChunk],
        draft_answer: str = "",
        intent: str = "unknown",
    ) -> Optional[str]:
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
        selected = (
            self.store.products_by_id.get(self.checkout_product_id)
            if self.checkout_product_id is not None
            else self.last_selected_product
        )
        state_block = (
            f"Intent atual: {intent}\n"
            f"Compra em andamento: {'sim' if self.checkout_active else 'não'}\n"
            f"Produto selecionado: {selected.name if selected else 'nenhum'}\n"
            f"Rascunho seguro do fluxo: {draft_answer or 'nenhum'}"
        )
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
            "Conversation state:\n"
            f"{state_block}\n\n"
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
    def _response_preserves_draft(
        answer: str,
        draft: str,
        store: CatalogStore,
    ) -> bool:
        """Reject a fluent rewrite when it drops important workflow facts."""

        if not draft:
            return True
        normalized_answer = normalize(answer)
        normalized_draft = normalize(draft)
        required_phrases = (
            "qual modelo",
            "nome completo",
            "telefone ou email",
            "7 dias corridos",
            "10 dias uteis",
            "30 dias corridos",
            "90 dias",
            "nao e cumulativo",
            "nao sao cumulativos",
            "nao sao compartilhados",
            "outras cidades",
            "cep",
            "peso",
            "dimensoes",
            "9h as 18h",
            "9h as 13h",
            "fechado",
            "24 horas uteis",
            "troca por preferencia",
            "embalagem original",
            "recebimento",
            "fabricante",
            "pedido ainda nao foi criado",
            "produto ainda nao foi criado",
            "rua 14 de maio",
        )
        for phrase in required_phrases:
            if phrase in normalized_draft and phrase not in normalized_answer:
                return False

        for amount in re.findall(r"r\$\s*[\d.,]+", normalized_draft):
            if amount not in normalized_answer:
                return False
        for stock in re.findall(r"\b\d+\s+unidade", normalized_draft):
            if stock not in normalized_answer:
                return False
        for quantity in re.findall(r"\b\d{1,2}x\b", normalized_draft):
            if quantity not in normalized_answer:
                return False
        for product in store.products:
            name = normalize(product.name)
            if len(name) >= 12 and name in normalized_draft and name not in normalized_answer:
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
        name = extract_checkout_name(message)
        if name:
            self.checkout_customer_name = name
        contact = extract_checkout_contact(message)
        if contact:
            self.checkout_contact = contact
        payment_method, installments = extract_checkout_payment(message)
        if payment_method:
            self.checkout_payment_method = payment_method
        if installments is not None:
            self.checkout_installments = installments
        if is_checkout_selection_confirmation(message) or named is not None:
            self.checkout_confirmed = True

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
        shortage = (
            f"Temos {product.stock_quantity} unidade(s) em estoque, então posso seguir "
            f"com {product.stock_quantity} ou buscar outro modelo."
            if self.checkout_quantity > product.stock_quantity
            else f"Temos {product.stock_quantity} unidade(s) disponíveis."
        )
        if not self.checkout_confirmed:
            return (
                f"Encontrei {item}; {shortage} Você quer esse modelo? Se sim, confirme a quantidade "
                "e envie nome completo, telefone ou e-mail e endereço de entrega.",
                grounding,
            )

        received_address = (
            f"Recebi o endereço de entrega: {self.checkout_address}. "
            if address
            else ""
        )
        missing: list[str] = []
        if not self.checkout_customer_name:
            missing.append("nome completo")
        if not self.checkout_contact:
            missing.append("telefone ou e-mail")
        if not self.checkout_address:
            missing.append("endereço de entrega")
        if not self.checkout_payment_method:
            missing.append("forma de pagamento")
        if missing:
            fields = ", ".join(missing[:-1]) + (
                f" e {missing[-1]}" if len(missing) > 1 else missing[0]
            )
            return (
                f"Perfeito! Selecionei {item}. {shortage} {received_address}"
                f"Para concluir, ainda preciso de {fields}. O pedido ainda não foi criado; "
                "ele só será registrado após essa confirmação.",
                grounding,
            )
        payment = self.checkout_payment_method
        if self.checkout_installments:
            payment += f" em {self.checkout_installments}x"
        return (
            f"Recebi os dados para {item}, em nome de {self.checkout_customer_name}, "
            f"contato {self.checkout_contact}, endereço {self.checkout_address} e pagamento em {payment}. "
            "O pedido ainda não foi criado; confirme se os dados estão corretos para registrarmos a compra.",
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

        if is_damage_report(message):
            answer = (
                "Sinto muito: entendi que o instrumento está com avaria. "
                f"Já estou acionando um atendente da equipe para assumir o seu caso "
                f"(protocolo {ticket['protocol']})."
            )
        else:
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
        if is_damage_report(message):
            answer += " Preserve a embalagem e, se possível, envie fotos do dano à equipe."
        answer += " O retorno acontece em até 24 horas úteis, no horário de atendimento."
        return answer, []

    def _category_overview(self) -> tuple[str, list[RetrievedChunk]]:
        counts: dict[int, int] = {}
        for item in self.store.products:
            if item.available:
                counts[item.category_id] = counts.get(item.category_id, 0) + 1
        rows = sorted(
            ((self.store.category_name(cid), total) for cid, total in counts.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if not rows:
            return "Não encontrei produtos disponíveis no catálogo agora.", []
        lines = ["Trabalhamos com estas famílias de instrumentos:"]
        lines += [f"- {name}: {total} modelo(s) disponíveis." for name, total in rows]
        lines.append("Me diga a família que te interessa e eu listo os modelos.")
        return "\n".join(lines), [
            RetrievedChunk(
                title="Categorias do catálogo",
                source_type="catalog_row",
                content="\n".join(f"{name}: {total} disponíveis" for name, total in rows),
                score=12.0,
                retrieval="structured",
            )
        ]

    def _catalog_answer(self, message: str) -> tuple[str, list[RetrievedChunk]]:
        category_id = self.store.category_id_for(message)
        max_price = parse_budget(message)
        product = self.store.best_product_match(message)
        words = keyword_set(message)
        query_norm = normalize(message)

        # Asked what the store sells, with no family named: answer with families.
        if (
            is_category_overview(message)
            and category_id is None
            and product is None
            and max_price is None
        ):
            self.last_selected_product = None
            return self._category_overview()

        brand_products = self.store.brand_products(
            message, category_id=category_id, max_price=max_price
        )
        brand_scope = self.store.brand_products(message, category_id=category_id)
        if (
            not brand_products
            and brand_scope
            and product is None
            and not query_model_numbers(message)
            and category_id is not None
            and max_price is not None
        ):
            brand = normalize(brand_scope[0].name).split()[0].title()
            label = self.store.category_name(category_id).lower()
            return (
                f"Não encontrei {label} da {brand} disponíveis até {format_brl(max_price)}.",
                [],
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
                # "Não, havia mais" only makes sense when the list was truncated;
                # after a complete answer the honest reply is "sim, são todos".
                if shown >= len(all_products):
                    return (
                        f"Sim, esses são todos os {len(all_products)} {label} "
                        f"disponíveis{budget}. Posso filtrar por marca, tipo ou faixa de preço.",
                        self.store.grounding_for_products(all_products),
                    )
                return (
                    f"Não. Encontrei {len(all_products)} {label} disponíveis{budget}. "
                    f"A lista anterior mostrava apenas {shown} opções; posso filtrar por marca, tipo ou faixa de preço.",
                    self.store.grounding_for_products(all_products[:shown]),
                )
            if re.search(r"\b(?:tem mais|mais|outros|restante|restantes|o que falta|faltam)\b", normalized_query):
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
                return f"Já mostrei todos os {label} disponíveis nesta busca ({len(all_products)} no total).", self.store.grounding_for_products(all_products)

        if category_id is not None:
            self.last_selected_product = None
            # Small filtered sets are complete answers; broad category searches
            # remain paginated to keep the chat readable.
            show_all_filtered = max_price is not None and len(all_products) <= 15
            products = all_products if show_all_filtered else all_products[:5]
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

    def _policy_answer(
        self, message: str, chunks: Optional[list[RetrievedChunk]] = None
    ) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.rag.search(message) if chunks is None else chunks
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
        elif is_damage_report(message) or words & {"defeito", "garantia"}:
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
    return bool(re.fullmatch(r"\s*(oi|oie+|ola|olá|bom dia|boa tarde|boa noite|hello|hi)[!. ]*", message, re.I))


def is_catalog_follow_up(message: str) -> bool:
    normalized = normalize(message).strip("?!., ")
    return bool(
        re.search(r"\b(?:todos|todas|cada|lista completa|mais modelos?|so tem esses|so esses|tem mais|tem outros|sao todos|sao esses|apenas esses|somente esses|restante|restantes|o que falta|faltam|outros|outras)\b", normalized)
        or re.search(r"\b(?:e\s+)?(?:quais|qual|que|mostre|mostrar|ver|lista(?:r)?)\s+(?:mais|outros|outras|restantes)\b", normalized)
        or re.fullmatch(r"e(?: da| do| das| dos| a| o)? [a-z0-9-]+", normalized)
    )

def is_contextual_follow_up(message: str) -> bool:
    normalized = normalize(message).strip("?!., ")
    return bool(
        re.match(r"^(?:e|mas|tambem)\b", normalized)
        or normalized in {"como funciona", "qual prazo", "e depois", "e nesse caso"}
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


def extract_checkout_name(message: str) -> Optional[str]:
    """Extract a customer name without treating an arbitrary product sentence as one."""

    address_match = re.search(
        r"\b(?:rua|avenida|av|alameda|travessa|estrada|rodovia|pra[cç]a)\b",
        message,
        re.IGNORECASE,
    )
    before_address = message[:address_match.start()] if address_match else message
    explicit = re.search(
        r"\b(?:meu\s+)?nome(?:\s+completo)?\s*[:\-]?\s*([^,;]+)",
        before_address,
        re.IGNORECASE,
    )
    if explicit:
        candidate = " ".join(explicit.group(1).split()).strip()
        if candidate:
            return candidate

    # Implicit names are accepted only in a comma-separated data message such as
    # "Sim, 1 apenas, Atila Carlos, 67...".
    if "," not in before_address:
        return None
    for part in before_address.split(","):
        candidate = re.sub(
            r"^\s*(?:sim|isso|confirmo|pode ser)\s*[:\-]?\s*",
            "",
            part,
            flags=re.IGNORECASE,
        ).strip()
        if not candidate or re.search(
            r"\d|@|\b(?:telefone|celular|fone|pagarei|pagar|credito|cartao|pix|boleto|debito)\b",
            candidate,
            re.IGNORECASE,
        ):
            continue
        words = re.findall(r"[A-Za-zÀ-ÿ]+", candidate)
        if len(words) >= 2:
            return " ".join(words)
    return None


def extract_checkout_contact(message: str) -> Optional[str]:
    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", message)
    if email:
        return email.group(0)
    phone = re.search(r"\+?\d(?:[\d ()-]{7,})\d", message)
    return phone.group(0).strip() if phone else None


def extract_checkout_payment(message: str) -> tuple[Optional[str], Optional[int]]:
    normalized = normalize(message).replace("´", "")
    labels = (
        (r"cre\s*dito", "crédito"),
        (r"cartao", "cartão"),
        (r"debito", "débito"),
        (r"pix", "PIX"),
        (r"boleto", "boleto"),
    )
    method = next(
        (label for token, label in labels if re.search(rf"\b{token}\b", normalized)),
        None,
    )
    installments = None
    if method:
        match = re.search(r"\b(\d{1,2})\s*x\b", normalized)
        if match:
            installments = int(match.group(1))
    return method, installments


# Spelled-out counts customers actually use; digits cover the rest.
QUANTITY_WORDS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}


def extract_quantity(message: str) -> Optional[int]:
    """Read units while keeping credit-card installments out of the quantity."""

    normalized = normalize(message)
    payment_installment = bool(
        re.search(r"\b(?:credito|cartao|parcel|pagarei|pagamento|vezes)\b", normalized)
        and re.search(r"\b\d{1,2}\s*x\b", normalized)
    )
    unit_pattern = (
        r"\b(\d{1,3})\s*(?:unidade|unidades|peca|pecas|item|itens)\b"
        if payment_installment
        else r"\b(\d{1,3})\s*(?:unidade|unidades|peca|pecas|item|itens|x)\b"
    )
    match = re.search(unit_pattern, normalized) or re.search(
        r"\b(\d{1,2})\s*(?:apenas|so|somente)\b", normalized
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


def is_checkout_selection_confirmation(message: str) -> bool:
    """Identify a confirmation/data turn without confusing it with a new query."""

    normalized = normalize(message).strip("?!., ")
    if is_checkout_confirmation(message):
        return True
    if not re.match(r"^(?:sim|isso|confirmo|pode ser)\b", normalized):
        return bool(
            extract_quantity(message)
            or extract_delivery_address(message)
            or extract_checkout_name(message)
            or extract_checkout_contact(message)
            or extract_checkout_payment(message)[0]
        )
    method, _ = extract_checkout_payment(message)
    return bool(
        extract_quantity(message)
        or extract_delivery_address(message)
        or extract_checkout_name(message)
        or extract_checkout_contact(message)
        or method
    )


def is_checkout_continuation(message: str, store: CatalogStore) -> bool:
    normalized = normalize(message)
    if extract_delivery_address(message) or is_checkout_confirmation(message):
        return True
    if re.search(
        r"\b(?:como faco|como prossigo|como continuo|o que faco agora|proximos? passos?|"
        r"pode ser|vou ficar com|quero esse|quero essa|escolho|prefiro)\b",
        normalized,
    ):
        return True
    # Quantity and payment-only replies must stay in the active checkout.
    if extract_quantity(message) is not None or extract_checkout_payment(message)[0] is not None:
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

def is_damage_report(message: str) -> bool:
    """Detect a purchased instrument reported as broken or otherwise damaged."""

    normalized = normalize(message)
    damage = re.search(
        r"\b(?:quebrad[oa]|danificad[oa]|avariad[oa]|rachad[oa]|trincad[oa]|"
        r"amassad[oa]|solt[oa]|quebrou)\b",
        normalized,
    )
    if not damage:
        return False
    context = keyword_set(message) & {
        "produto", "instrumento", "bateria", "violao", "guitarra", "baixo",
        "teclado", "piano", "ukulele", "comprei", "recebi", "compra", "pedido",
    }
    part = re.search(r"\b(?:perna|parte|peca|suporte|pedal|casco|pele)\b", normalized)
    return bool(context or part)


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
    return is_damage_report(message) or any(
        re.search(pattern, normalized) for pattern in HANDOFF_PATTERNS
    )


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
    location_signal = bool(
        re.search(r"\b(?:onde|qual)\b.{0,25}\b(?:fica|endereco|localizacao)\b", normalize(message))
        or re.search(r"\b(?:onde fica|endereco da|localizacao da)\s+(?:a\s+)?loja\b", normalize(message))
    )
    policy_signal = bool(
        location_signal
        or is_damage_report(message)
        or is_exchange_request(message)
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
    inventory_catalog_signal = bool(
        (store.category_id_for(message) is not None or store.best_product_match(message) is not None)
        and re.search(
            r"\b(?:pronta[- ]entrega|em estoque|disponibilidade|disponivel|disponíveis)\b",
            normalize(message),
        )
    )
    if inventory_catalog_signal:
        return "catalog"
    if policy_signal:
        return "policy"

    catalog_signal = bool(
        is_category_overview(message)
        or words & {
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



__all__ = [
    "ACCESSORY_TERMS",
    "CHECKOUT_PATTERNS",
    "DAMAGE_PATTERNS",
    "HANDOFF_PATTERNS",
    "HANDOFF_TRIGGER_LABELS",
    "StoreAgent",
    "extract_checkout_contact",
    "extract_checkout_name",
    "extract_checkout_payment",
    "extract_delivery_address",
    "extract_exchange_date",
    "extract_order_reference",
    "extract_quantity",
    "is_catalog_follow_up",
    "is_checkout_confirmation",
    "is_checkout_continuation",
    "is_checkout_request",
    "is_contextual_follow_up",
    "is_exchange_continuation",
    "is_damage_report",
    "is_exchange_request",
    "is_greeting",
    "is_injection",
    "needs_human_handoff",
    "route_message",
]