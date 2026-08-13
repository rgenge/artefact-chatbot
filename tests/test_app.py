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

    def test_live_catalog_list_cannot_be_summarized(self):
        agent = StoreAgent(DATA, use_llm=True)
        agent.client.api_key = "local-test-key"
        agent._gemini_answer = lambda message, grounding: "Yamaha C40 e Yamaha F310."
        answer = agent.handle("Quais violões Yamaha estão disponíveis?")
        self.assertIn("Encontrei 7 violões", answer)
        self.assertIn("Yamaha CG162S Nylon Natural", answer)
        self.assertIn("Yamaha NTX1 Elétrico Nylon Natural", answer)

    def test_natural_catalog_follow_up_lists_previous_results(self):
        agent = StoreAgent(DATA, use_llm=False)
        agent.handle("Quais violões Yamaha estão disponíveis?")
        answer = agent.handle("Só tem esses?")
        self.assertIn("Encontrei 7 violões", answer)
        self.assertIn("Yamaha CG162S Nylon Natural", answer)

    def test_brand_typo_is_corrected_and_lists_all_models(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Quais violões yahama tem?")
        self.assertIn("Encontrei 7 violões da Yamaha", answer)
        self.assertIn("Yamaha NTX1 Elétrico Nylon Natural", answer)
        self.assertIn("Yamaha CG162S Nylon Natural", answer)

    def test_generic_category_reports_total_and_preview(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Quais violões tem?")
        self.assertIn("Encontrei 33 violões disponíveis", answer)
        self.assertIn("Mostrando 5 opções", answer)

    def test_generic_follow_up_explains_preview_and_more_results(self):
        agent = StoreAgent(DATA, use_llm=False)
        agent.handle("Quais violões tem?")
        answer = agent.handle("Só esses ?")
        self.assertIn("Não.", answer)
        self.assertIn("33 violões", answer)
        self.assertIn("apenas 5 opções", answer)

        more = agent.handle("Tem mais?")
        self.assertIn("Sim, há mais violões", more)
        self.assertNotIn("Tagima Memphis AC-39 Nylon Natural", more)

    def test_brand_follow_up_preserves_category_context(self):
        agent = StoreAgent(DATA, use_llm=False)
        agent.handle("Quais violões tem?")
        answer = agent.handle("E da Tagima ?")
        self.assertIn("Encontrei 6 violões da Tagima", answer)
        self.assertIn("Tagima Vegas Elétrico Aço Natural", answer)

    def test_unknown_question_does_not_invent_personal_life(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("O que tu faz no final de semana?")
        self.assertIn("instrumentos", answer.lower())
        self.assertNotIn("organizar catálogo", answer.lower())
        self.assertNotIn("final de semana", answer.lower())

    def test_black_friday_distinguishes_policy_from_active_promotions(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Tem algum com Black Friday?")
        self.assertIn("novembro", answer)
        self.assertIn("15% a 30%", answer)
        self.assertIn("não há uma promoção ativa identificada como Black Friday", answer)

    def test_accessory_scope_is_explicit(self):
        agent = StoreAgent(DATA, use_llm=False)
        answer = agent.handle("Vocês vendem cabos e palhetas?")
        self.assertIn("exclusivamente com instrumentos musicais", answer)


if __name__ == "__main__":
    unittest.main()


