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

    def ask(self, message, customer_id=None):
        payload = json.dumps(
            {"message": message, "customer_id": customer_id, "conversation_id": "frontend-history-test", "handoff": False},
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


if __name__ == "__main__":
    unittest.main()