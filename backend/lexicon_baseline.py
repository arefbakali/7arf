"""
7arf - Référence NON neuronale (dictionnaire + règles).
Sert de repli (« mode démo ») quand PyTorch ou le modèle est indisponible, et de point de
comparaison dans l'interface si le modèle A est absent.
"""
import json
import os
from collections import Counter

# table de règles simple (digrammes d'abord)
DIGRAPHS = {"ch": "ش", "sh": "ش", "dh": "ذ", "th": "ث", "gh": "غ", "kh": "خ", "ou": "و", "oo": "و",
            "ee": "ي", "ei": "ي", "ai": "اي"}
CHARS = {"a": "ا", "b": "ب", "t": "ت", "j": "ج", "7": "ح", "5": "خ", "d": "د", "r": "ر", "z": "ز",
         "s": "س", "9": "ق", "q": "ق", "f": "ف", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه",
         "8": "ه", "w": "و", "o": "و", "u": "و", "y": "ي", "i": "ي", "e": "", "é": "ي", "è": "ي",
         "3": "ع", "2": "ء", "g": "ڨ", "p": "ب", "v": "ف", "x": "كس", "c": "ك", "6": "ط"}


def rule_translit(w):
    out, i = [], 0
    while i < len(w):
        if w[i:i + 2] in DIGRAPHS:
            out.append(DIGRAPHS[w[i:i + 2]]); i += 2; continue
        c = w[i]
        out.append(CHARS.get(c, c if not c.isalpha() else ""))
        i += 1
    s = "".join(out)
    return s or w


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class LexiconTransliterator:
    """Référence non neuronale réutilisable (sert aussi de mode démo dans l'interface)."""

    def __init__(self, data_dir="data/lexicon"):
        self.best, self.french = {}, set()
        lp = os.path.join(data_dir, "train_lexicon.json")
        if os.path.exists(lp):
            lex = json.load(open(lp, encoding="utf-8"))
            self.best = {s: Counter(t).most_common(1)[0][0] for s, t in lex.items()}
        tp = os.path.join(data_dir, "chunk", "train.jsonl")
        if os.path.exists(tp):
            self.french = {s for e in _read_jsonl(tp) for s, c in zip(e["src_tokens"], e["classes"])
                           if c == "foreign"}

    def token(self, tok):
        if tok in self.best:
            return self.best[tok]
        if tok in self.french:
            return tok
        return rule_translit(tok)

    def __call__(self, text):
        return " ".join(self.token(t) for t in text.split())
