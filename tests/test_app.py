import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import StoreAgent


DATA = ROOT / "data"


class StoreAgentTests(unittest.TestCase):
    def test_catalog_budget_uses_structured_stock_and_price(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Quais opções de violões disponíveis custando até R$1000?")
        self.assertIn("Tagima Memphis AC-39 Nylon Natural", answer)
        self.assertIn("R$ 429,90", answer)
        self.assertNotIn("Giannini GF-3D", answer)

    def test_product_price_and_stock_are_exact(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Quanto custa o Takamine GD20?")
        self.assertIn("R$ 2.199,00", answer)
        self.assertIn("5 unidade(s)", answer)

    def test_policy_retrieval_applies_return_rule(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Me arrependi da minha compra, posso devolver meu pedido?")
        self.assertIn("7 dias corridos", answer)
        self.assertIn("10 dias úteis", answer)
        self.assertIn("Manual de Políticas", answer)

    def test_order_requires_customer_context(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Qual o status do pedido 8?")
        self.assertIn("proteger seus dados", answer)

    def test_order_uses_customer_context_and_joined_items(self):
        agent = StoreAgent(DATA, customer_id=2, use_llm=False)
        answer = agent.handle("Qual o status do pedido 8?")
        self.assertIn("enviado", answer)
        self.assertIn("rastreamento", answer.lower())
        self.assertIn("Kala KA-C Concert Mogno", answer)

    def test_brand_query_lists_all_models_and_follow_up(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Quais violões Yamaha tem?")
        for model in (
            "Yamaha C40 Nylon Natural",
            "Yamaha C70 Nylon Natural",
            "Yamaha F310 Aço Natural",
            "Yamaha FG800 Dreadnought Natural",
            "Yamaha NTX1 Elétrico Nylon Natural",
            "Yamaha APX600 Elétrico Aço Preto",
            "Yamaha CG162S Nylon Natural",
        ):
            self.assertIn(model, answer)
        self.assertIn("Encontrei 7 violões", answer)
        follow_up = agent.handle("Quero saber todos")
        self.assertIn("Yamaha CG162S Nylon Natural", follow_up)

    def test_accessory_scope_is_explicit(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Vocês vendem cabos e palhetas?")
        self.assertIn("exclusivamente com instrumentos musicais", answer)


if __name__ == "__main__":
    unittest.main()


