"""7arf - Vocabulaire de caractères (sans dépendance PyTorch)."""
import json
from collections import Counter

PAD, UNK, SOS, EOS = "<PAD>", "<UNK>", "<SOS>", "<EOS>"
SPECIALS = [PAD, UNK, SOS, EOS]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


class CharVocab:
    """Tokenisation au niveau caractère. L'espace est un symbole comme un autre."""

    def __init__(self, chars=()):
        self.itos = list(SPECIALS) + [c for c in chars if c not in SPECIALS]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        cnt = Counter(c for t in texts for c in t)
        chars = sorted(c for c, n in cnt.items() if n >= min_freq)
        return cls(chars)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, add_sos_eos=False, max_len=None):
        ids = [self.stoi.get(c, UNK_ID) for c in text]
        if max_len is not None:
            ids = ids[:max_len]  # troncature
        if add_sos_eos:
            ids = [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids, strip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID and strip_special:
                break
            if strip_special and i in (PAD_ID, SOS_ID):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "")
        return "".join(c if c != UNK else "�" for c in out)

    def unk_rate(self, texts):
        tot = sum(len(t) for t in texts)
        unk = sum(1 for t in texts for c in t if c not in self.stoi)
        return unk / max(tot, 1)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            itos = json.load(f)
        v = cls()
        v.itos, v.stoi = itos, {c: i for i, c in enumerate(itos)}
        return v
