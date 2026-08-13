"""Dependency-free local browser UI for the store agent."""
from __future__ import annotations
import html, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from app import HANDOFF_TRIGGER_LABELS, StoreAgent

PAGE = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Empório da Música</title><style>body{margin:0;background:#eef6f0;color:#18221d;font:16px system-ui,sans-serif}main{max-width:850px;margin:auto;padding:32px 16px}h1{font:700 42px Georgia;margin:0 0 20px}.tag{color:#0d6849;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase}.box{background:white;border:1px solid #dce5df;border-radius:22px;overflow:hidden;box-shadow:0 15px 40px #173a2912}.top{background:#f7faf7;border-bottom:1px solid #dce5df;padding:16px}.top small,footer{color:#6c766f}.messages{min-height:390px;max-height:60vh;overflow:auto;padding:20px}.msg{display:flex;margin:12px 0}.user{justify-content:flex-end}.bubble{max-width:78%;padding:12px 15px;border-radius:17px;white-space:pre-wrap}.assistant .bubble{background:#f7faf7;border:1px solid #dce5df}.user .bubble{background:#0d6849;color:white}.suggest{display:flex;gap:8px;flex-wrap:wrap;padding:0 20px 16px}.suggest button{border:1px solid #dce5df;background:white;border-radius:20px;padding:8px;color:#0d6849;cursor:pointer}.suggest button.handoff{margin-left:auto;border-color:#e2c9a8;color:#96601b}.suggest button.handoff[disabled]{opacity:.45;cursor:not-allowed}.hand{background:#fffdf7;border-bottom:1px solid #dce5df;padding:12px 16px;font-size:14px}.hand .sw{display:flex;gap:9px;align-items:center;font-weight:600;color:#96601b;cursor:pointer}.hand summary{cursor:pointer;color:#6c766f;margin-top:7px}.hand ul{margin:8px 0 0;padding-left:19px;color:#6c766f}.hand li{margin:4px 0}.hand em{color:#96601b;font-style:normal}form{display:flex;gap:8px;border-top:1px solid #dce5df;padding:16px}input,form button{font:inherit;border-radius:11px;padding:11px;border:1px solid #dce5df}#message{flex:1}#customer{width:120px}form button{background:#0d6849;color:white;cursor:pointer}footer{text-align:center;font-size:12px;margin-top:12px}@media(max-width:600px){h1{font-size:34px}.bubble{max-width:90%}form{flex-wrap:wrap}#message{flex-basis:100%}}</style><main><p class="tag">Atendimento inteligente</p><h1>Empório da Música</h1><section class="box"><div class="top">🟢 <b>Assistente da loja</b> <small> · Gemini + RAG híbrido</small></div><div class="hand"><label class="sw"><input type="checkbox" id="handoff" checked> Encaminhar para um atendente humano</label><details><summary>O que aciona um atendente?</summary><ul><!--TRIGGERS--></ul></details></div><div id="messages" class="messages"><div class="msg assistant"><div class="bubble">Olá! Posso ajudar com instrumentos, preços, estoque, promoções, pedidos e políticas da loja.</div></div></div><div class="suggest"><button data-q="Quais violões estão disponíveis até R$ 2.000?">Ver violões</button><button data-q="Quais promoções estão ativas?">Promoções</button><button data-q="Qual é a política de devolução?">Devolução</button><button class="handoff" data-q="Meu violão está atrasado">Falar com atendente</button></div><form id="chat"><input id="customer" type="number" min="1" placeholder="Cliente"><input id="message" required autocomplete="off" placeholder="Digite sua pergunta..."><button>Enviar</button></form></section><footer>Catálogo estruturado + manual em RAG por chunks.</footer></main><script>const box=document.querySelector('#messages'),form=document.querySelector('#chat'),input=document.querySelector('#message'),customer=document.querySelector('#customer'),send=form.querySelector('button'),handoff=document.querySelector('#handoff'),chip=document.querySelector('.suggest button.handoff');handoff.onchange=()=>{chip.disabled=!handoff.checked};function add(t,c){let r=document.createElement('div'),b=document.createElement('div');r.className='msg '+c;b.className='bubble';b.textContent=t;r.append(b);box.append(r);box.scrollTop=box.scrollHeight}async function ask(t){t=(t||input.value).trim();if(!t)return;add(t,'user');input.value='';send.disabled=true;try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t,customer_id:customer.value||null,handoff:handoff.checked})});let d=await r.json();add(d.answer||d.error||'Não foi possível responder.','assistant')}catch(e){add('Não foi possível conectar ao agente local.','assistant')}send.disabled=false;input.focus()}form.onsubmit=e=>{e.preventDefault();ask()};document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>ask(b.dataset.q));</script>'''

# The trigger list comes from app.py so it cannot drift from the real rules.
PAGE = PAGE.replace('<!--TRIGGERS-->', ''.join(
    f'<li><b>{html.escape(label)}</b> — <em>“{html.escape(example)}”</em></li>'
    for label, example in HANDOFF_TRIGGER_LABELS
))

def serve(data_dir: Path, host: str = "127.0.0.1", port: int = 8000, use_llm: bool = True, model: str | None = None) -> None:
    agents = {}; lock = threading.Lock()
    def get_agent(customer_id):
        key = str(customer_id or "anonymous")
        with lock:
            if key not in agents: agents[key] = StoreAgent(data_dir, customer_id=customer_id, use_llm=use_llm, model=model)
            return agents[key]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass
        def do_GET(self):
            if urlparse(self.path).path != '/': self.send_error(404); return
            body=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            if urlparse(self.path).path != '/api/chat': self.send_error(404); return
            try:
                payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0')))); message=str(payload.get('message','')).strip(); raw=payload.get('customer_id'); customer_id=int(raw) if raw not in (None,'') else None
                if not message: raise ValueError('message is required')
                agent=get_agent(customer_id); agent.handoff_enabled=bool(payload.get('handoff',True))
                result={'answer':agent.handle(message)}; status=200
            except (ValueError,TypeError,json.JSONDecodeError) as exc: result={'error':str(exc)}; status=400
            except Exception: result={'error':'Agent error'}; status=500
            body=json.dumps(result,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    server=ThreadingHTTPServer((host,port),Handler); print(f'UI disponível em http://{host}:{port}')
    try: server.serve_forever()
    except KeyboardInterrupt: print('\nEncerrando.')
    finally: server.server_close()
if __name__ == '__main__': serve(Path(__file__).resolve().parent/'data')
