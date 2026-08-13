import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PORT = 18765
BASE_URL = f"http://127.0.0.1:{PORT}"


class FrontendChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen(
            [
                sys.executable,
                "app.py",
                "--web",
                "--no-llm",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urlopen(f"{BASE_URL}/", timeout=0.25) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError):
                time.sleep(0.1)
        cls.process.terminate()
        cls.process.wait(timeout=5)
        raise RuntimeError("The local frontend did not start")

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5)

    def ask(self, message, customer_id=None, conversation_id="frontend-history-test", handoff=False):
        payload = json.dumps(
            {"message": message, "customer_id": customer_id, "conversation_id": conversation_id, "handoff": handoff},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
        self.assertNotIn("error", body)
        return body["answer"]

    def test_frontend_keeps_one_chat_history_across_api_requests(self):
        with urlopen(f"{BASE_URL}/", timeout=10) as response:
            page = response.read().decode("utf-8")
        self.assertIn("Empório da Música", page)
        self.assertIn("Gemini + RAG híbrido", page)

        self.assertIn("Empório da Música", self.ask("oie"))

        first = self.ask("quero violã oaté 900 reais")
        self.assertIn("Encontrei 11 violões disponíveis até R$ 900,00", first)

        low = self.ask("e violão até 300 reais")
        self.assertIn("Não encontrei violões disponíveis até R$ 300,00", low)

        middle = self.ask("e até 600 reais ?", customer_id=2)
        self.assertIn("Encontrei 5 violões disponíveis até R$ 600,00", middle)
        self.assertIn("Yamaha C40 Nylon Natural", middle)
        self.assertNotIn("Pode reformular", middle)

        policy = self.ask("qual sua politica de deveolução ?")
        self.assertIn("7 dias corridos", policy)

        payment = self.ask("e pra pagar parcelado ?")
        self.assertIn("12x sem juros", payment)
        self.assertIn("parcela mínima de R$ 100,00", payment)

    def test_real_api_handles_policy_and_checkout_context(self):
        conversation = "frontend-real-hybrid-flow"

        location = self.ask("Onde fica loja?", conversation_id=conversation)
        self.assertIn("Rua 14 de Maio", location)
        self.assertNotIn("Pode reformular", location)

        self.ask("Se eu quiser finalizar uma compra como faz?", conversation_id=conversation)
        catalog = self.ask("quero uma guitarra le paul", conversation_id=conversation)
        self.assertIn("Ibanez RG550 Genesis Collection", catalog)
        self.assertIn("Gibson Les Paul Standard", catalog)

        selected = self.ask("ah pode ser a ibanez", conversation_id=conversation)
        self.assertIn("Selecionei 1x Ibanez RG550 Genesis Collection", selected)
        self.assertNotIn("Pode reformular", selected)

        next_step = self.ask("como faço?", conversation_id=conversation)
        self.assertIn("Ibanez RG550 Genesis Collection", next_step)
        self.assertIn("nome completo", next_step)
        self.assertNotIn("Pode reformular", next_step)

    def test_real_api_never_lists_catalog_for_broken_instrument(self):
        conversation = "frontend-damaged-instrument"

        self.ask("Oie to com problema", conversation_id=conversation, handoff=True)
        answer = self.ask(
            "comprei uma bateria ai e está com perna quebrada",
            conversation_id=conversation,
            handoff=True,
        )

        self.assertIn("avaria", answer.lower())
        self.assertIn("protocolo", answer.lower())
        self.assertNotIn("Encontrei 3 baterias", answer)

    def test_real_api_catalog_context_survives_policy_question(self):
        conversation = "frontend-catalog-after-policy"
        self.ask("Qual o prazo de entrega?", conversation_id=conversation)
        answer = self.ask("O que você tem pra vender?", conversation_id=conversation)
        self.assertIn("famílias de instrumentos", answer)
        self.assertIn("Violões: 33 modelo(s)", answer)
        self.assertNotIn("orientação no manual", answer.lower())

    def test_real_api_accumulates_checkout_fields_and_installments(self):
        conversation = "frontend-checkout-accumulation"
        self.ask("Quais violões até 500 reais?", conversation_id=conversation)
        self.ask("Quero um Tagima Memphis então", conversation_id=conversation)
        asked = self.ask("Quero comprar ele", conversation_id=conversation)
        self.assertIn("Você quer esse modelo?", asked)

        first = self.ask(
            "Sim, 1 apenas, atila carlos, 67 99999999, rua teste, 55 7",
            conversation_id=conversation,
        )
        self.assertIn("forma de pagamento", first)
        self.assertIn("1x Tagima Memphis AC-39 Nylon Natural", first)

        second = self.ask(
            "nome: atila carlos, telefone 67 99961-5555, pagarei em crédito 2 x",
            conversation_id=conversation,
        )
        self.assertIn("1x Tagima Memphis AC-39 Nylon Natural", second)
        self.assertNotIn("2x Tagima Memphis AC-39 Nylon Natural", second)
        self.assertIn("crédito em 2x", second)
        self.assertIn("pedido ainda não foi criado", second)
    def test_real_api_completes_short_choice_and_split_checkout_data(self):
        conversation = "frontend-yamaha-c40-final-checkout"
        listing = self.ask("hum e violão normal da yamaha?", conversation_id=conversation)
        self.assertIn("Encontrei 7 violões da Yamaha", listing)

        choice = self.ask("Quero o C40", conversation_id=conversation)
        self.assertIn("Yamaha C40 Nylon Natural: R$ 599,90", choice)
        self.assertNotIn("Pode reformular", choice)

        selected = self.ask("Quero comprar o C40", conversation_id=conversation)
        self.assertIn("Perfeito! Selecionei 1x Yamaha C40 Nylon Natural", selected)
        self.assertIn("nome completo", selected)

        data = self.ask(
            "Atila, 679999999, tEste@teste.com.br, rua teste, 556",
            conversation_id=conversation,
        )
        self.assertIn("forma de pagamento", data)
        self.assertNotIn("nome completo", data)

        installment = self.ask(
            "Atila da Silva, pagamento em 3 x",
            conversation_id=conversation,
        )
        self.assertIn("crédito em 3x", installment)
        self.assertIn("pedido ainda não foi criado", installment)
        self.assertNotIn("Aceitamos PIX", installment)

        pix = self.ask("pode ser pix", conversation_id=conversation)
        self.assertIn("pagamento em PIX", pix)
        self.assertNotIn("em 3x", pix)
        self.assertNotIn("Pode reformular", pix)
if __name__ == "__main__":
    unittest.main()
