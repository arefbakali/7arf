"""
7arf - Serveur web (bibliothèque standard Python uniquement) : interface + API.

  python -m backend.server                      # http://localhost:8000
  python -m backend.server --port 8000 --attention models/C_bi_attn_s2 --baseline models/A_noattn_s1

API (JSON) :
  GET  /api/info                -> mode (neural/demo), modèles chargés, alphabet appris
  POST /api/transliterate       {"text": "...", "beam": 0|5, "compare": bool}
  POST /api/ocr                 {"image": "<data URL ou base64>"} -> lignes lues dans la capture
"""
import argparse
import base64
import json
import mimetypes
import os
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from .engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
MAX_BODY = 12 * 1024 * 1024
ENGINE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "7arf/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("Image trop lourde (12 Mo maximum).")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/info":
            return self._json(ENGINE.info())
        if path == "/":
            path = "/index.html"
        f = os.path.normpath(os.path.join(APP_DIR, path.lstrip("/")))
        if not f.startswith(APP_DIR) or not os.path.isfile(f):
            return self._json({"error": "Page introuvable."}, 404)
        data = open(f, "rb").read()
        self.send_response(200)
        ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            req = self._body()
            if self.path == "/api/transliterate":
                text = str(req.get("text", ""))[:4000]
                if not text.strip():
                    return self._json({"error": "Écrivez au moins un mot en Arabizi."}, 400)
                return self._json(ENGINE.transliterate(text, beam=int(req.get("beam", 0)),
                                                       compare=bool(req.get("compare", False))))
            if self.path == "/api/ocr":
                img = str(req.get("image", ""))
                if "," in img[:100]:
                    img = img.split(",", 1)[1]
                if not img:
                    return self._json({"error": "Aucune image reçue."}, 400)
                from .ocr import ocr_lines
                try:
                    return self._json(ocr_lines(base64.b64decode(img)))
                except ImportError:
                    return self._json({"error": "OCR indisponible : installez tesseract-ocr et pytesseract."}, 500)
            return self._json({"error": "Adresse inconnue."}, 404)
        except ValueError as e:
            return self._json({"error": str(e)}, 413)
        except Exception as e:
            return self._json({"error": f"Erreur du serveur : {e}"}, 500)


def run(port=8000, attention=None, baseline=None, models=os.path.join(ROOT, "models"),
        data=os.path.join(ROOT, "data", "lexicon"), results=os.path.join(ROOT, "results"), runs=os.path.join(ROOT, "runs"),
        host="127.0.0.1"):
    global ENGINE
    ENGINE = Engine(attention, baseline, models, data, results, runs)
    info = ENGINE.info()
    print(f"7arf : mode {info['mode']}"
          + (f" | modèle {info['attention_model']['run']}" if info["attention_model"] else "")
          + (f" | {info['note']}" if info["note"] else ""))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Interface : http://localhost:{port}")
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--attention", default=None,
                    help="dossier du modèle à attention (défaut : meilleur C_bi_attn* sur la validation)")
    ap.add_argument("--baseline", default=None, help="dossier du modèle sans attention (défaut : A_noattn*)")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "lexicon"))
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour exposer sur le réseau local")
    a = ap.parse_args()
    run(a.port, a.attention, a.baseline, a.models, a.data, host=a.host).serve_forever()


if __name__ == "__main__":
    main()
