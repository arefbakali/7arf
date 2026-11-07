"""7arf - Normalisation du texte Arabizi (identique à celle de l'entraînement)."""
import re
import unicodedata

USER = "§"  # caractère réservé : remplace les pseudonymes "m5abbiNNN" / "مخبّيNNN"
_USER_SRC = re.compile(r"m5abbi\d*")
_USER_TGT = re.compile(r"مخب+ّ?ي\d*")
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿ـ]")  # ZW*, LRM/RLM, tatweel
_REPEAT = re.compile(r"(.)\1{3,}")  # 4 répétitions ou plus


def normalize_src(s: str) -> str:
    """Arabizi : NFKC, minuscules, pseudonymes -> §, répétitions écrasées à 3."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _INVISIBLE.sub("", s)
    s = _USER_SRC.sub(USER, s)
    s = _REPEAT.sub(lambda m: m.group(1) * 3, s)
    return s.strip()
