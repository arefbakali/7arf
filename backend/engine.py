"""
7arf - Moteur utilisé par l'interface web.

- Si PyTorch et un modèle entraîné sont disponibles : notre encodeur-décodeur (from scratch).
- Sinon : mode démo avec la référence non neuronale (dictionnaire + règles), clairement signalé.

Une ligne est normalisée comme à l'entraînement, puis découpée en morceaux de <= 6 tokens
(la taille vue pendant l'entraînement) ; chaque morceau est translittéré, puis on recolle.
"""
import glob
import json
import os

from .text import normalize_src
from .lexicon_baseline import LexiconTransliterator

K = 6


def _best_run(models_dir, prefix, runs_dir="runs"):
    """Dossier du checkpoint `prefix*` au meilleur CER de VALIDATION (jamais choisi sur le test).
    Aucun modèle avant l'entraînement : retourne None."""
    best, best_cer = None, float("inf")
    for ck in sorted(glob.glob(os.path.join(models_dir, f"{prefix}*", "best.pt"))):
        run = os.path.dirname(ck)
        cer = float("inf")
        # Le notebook écrit train_summary.json à côté de best.pt (models/<run>/) ;
        # l'ancien emplacement runs/<run>/ reste accepté.
        for summary in (os.path.join(run, "train_summary.json"),
                        os.path.join(runs_dir, os.path.basename(run), "train_summary.json")):
            try:
                cer = json.load(open(summary, encoding="utf-8"))["best_val_cer"]
                break
            except (OSError, KeyError, ValueError):
                continue
        if best is None or cer < best_cer:
            best, best_cer = run, cer
    return best


class Engine:
    def __init__(self, attention_run=None, baseline_run=None, models_dir="models", data_dir="data/lexicon",
                 results_dir="results", runs_dir="runs",
                 device=None):
        self.lexicon = LexiconTransliterator(data_dir)
        self.models = {}
        self.alphabet = None
        self.error = None
        attention_run = attention_run or _best_run(models_dir, "C_bi_attn", runs_dir) or _best_run(models_dir, "B_attn", runs_dir)
        baseline_run = baseline_run or _best_run(models_dir, "A_noattn", runs_dir)
        try:
            import torch
            from .inference import load_model
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            for key, run in (("attention", attention_run), ("baseline", baseline_run)):
                if run and os.path.exists(os.path.join(run, "best.pt")):
                    m, sv, tv, args = load_model(run, self.device)
                    self.models[key] = {"model": m, "sv": sv, "tv": tv, "run": os.path.basename(run),
                                        "attention": m.decoder.use_attention,
                                        "bidirectional": m.config["bidirectional"]}
            if attention_run:
                ap = os.path.join(results_dir, os.path.basename(attention_run), "learned_alphabet.json")
                if os.path.exists(ap):
                    self.alphabet = json.load(open(ap, encoding="utf-8"))
        except ImportError:
            self.error = "PyTorch n'est pas installé : mode démo."
        except Exception as e:  # modèle illisible, etc.
            self.error = f"Modèle non chargé ({e}) : mode démo."
        if not self.models and not self.error:
            self.error = "Aucun modèle entraîné trouvé (lancez le notebook) : mode démo avec règles simples."

    @property
    def mode(self):
        return "neural" if "attention" in self.models else "demo"

    def info(self):
        def desc(k):
            if k not in self.models:
                return None
            m = self.models[k]
            return {"run": m["run"], "attention": m["attention"], "bidirectional": m["bidirectional"]}
        return {"mode": self.mode, "attention_model": desc("attention"), "baseline_model": desc("baseline"),
                "alphabet": self.alphabet, "note": self.error}

    def _neural(self, key, chunk, beam):
        from .inference import transliterate
        m = self.models[key]
        out, att = transliterate(m["model"], m["sv"], m["tv"], chunk, self.device, beam=beam)
        att_list = None
        if att is not None and len(out):
            att_list = [[round(float(v), 3) for v in row] for row in att[:len(out)].tolist()]
        return out, att_list

    def transliterate_line(self, text, beam=0, compare=False):
        x = normalize_src(text)
        tokens = x.split()
        chunks = [" ".join(tokens[i:i + K]) for i in range(0, len(tokens), K)] or [""]
        out_chunks, base_chunks = [], []
        for ch in chunks:
            if not ch:
                continue
            if self.mode == "neural":
                out, att = self._neural("attention", ch, beam)
            else:
                out, att = self.lexicon(ch), None
            out_chunks.append({"src": ch, "out": out, "att": att})
            if compare:
                if "baseline" in self.models:
                    base_chunks.append(self._neural("baseline", ch, beam)[0])
                elif self.mode == "neural":
                    base_chunks.append(self.lexicon(ch))
        res = {"input": text, "normalized": x, "output": " ".join(c["out"] for c in out_chunks),
               "chunks": out_chunks}
        if compare and base_chunks:
            res["baseline"] = " ".join(base_chunks)
            res["baseline_name"] = ("sans attention (A)" if "baseline" in self.models
                                    else "dictionnaire + règles")
        return res

    def transliterate(self, text, beam=0, compare=False):
        lines = [l for l in text.splitlines() if l.strip()]
        return {"mode": self.mode, "lines": [self.transliterate_line(l, beam, compare) for l in lines]}
