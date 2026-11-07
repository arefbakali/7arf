"""
7arf - Lecture de captures d'écran (OCR) avant la translittération.

L'OCR est une étape de PRÉTRAITEMENT : il transforme l'image en lignes de texte
Arabizi. La génération (Arabizi -> arabe) reste faite par notre encodeur-décodeur.
On utilise Tesseract (langues fra + eng, l'Arabizi étant écrit en lettres latines).

Prétraitement de l'image :
  - niveaux de gris ; inversion si l'écran est en mode sombre ;
  - agrandissement des petites captures (Tesseract préfère ~30 px par ligne) ;
  - contraste automatique.
Post-traitement :
  - regroupement des mots par ligne (bloc, paragraphe, ligne de Tesseract) ;
  - lignes de faible confiance ou ressemblant à une heure ("14:32", "lu 9:05") marquées
    comme à ignorer par défaut, l'utilisateur pouvant les rétablir dans l'interface.

  python -m src.ocr capture.png
"""
import io
import re
import sys

from PIL import Image, ImageOps, ImageStat

_TIME = re.compile(r"^(\w{0,4}\s*)?\d{1,2}\s*[:h.]\s*\d{2}(\s*[ap]\.?m\.?)?(\s*[✓✔vV/]+)?$", re.I)
_NOISE = re.compile(r"^[\W_]{0,3}$")


def preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    g = ImageOps.grayscale(img)
    if ImageStat.Stat(g).mean[0] < 110:          # mode sombre -> texte clair sur fond foncé
        g = ImageOps.invert(g)
    if g.width < 1400:                            # petites captures : on agrandit
        f = min(3.0, 1600 / g.width)
        g = g.resize((int(g.width * f), int(g.height * f)), Image.LANCZOS)
    return ImageOps.autocontrast(g, cutoff=1)


def ocr_lines(image_bytes: bytes, lang: str = "fra+eng", min_conf: float = 35.0):
    """Retourne {"lines": [{"text", "conf", "keep", "box"}], "width", "height"}."""
    import pytesseract
    from pytesseract import Output
    import os, shutil

    # Windows : Tesseract n'est pas toujours dans le PATH
    exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not shutil.which("tesseract") and os.path.exists(exe):
        pytesseract.pytesseract.tesseract_cmd = exe
    user_tessdata = os.path.join(os.path.expanduser("~"), "tessdata")
    if os.path.exists(os.path.join(user_tessdata, "fra.traineddata")):
        os.environ.setdefault("TESSDATA_PREFIX", user_tessdata)

    img = Image.open(io.BytesIO(image_bytes))
    w0, h0 = img.size
    g = preprocess(img)
    scale = g.width / w0
    data = pytesseract.image_to_data(g, lang=lang, config="--psm 4", output_type=Output.DICT)

    groups = {}
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        gr = groups.setdefault(key, {"words": [], "confs": [], "x0": 1e9, "y0": 1e9, "x1": 0, "y1": 0})
        gr["words"].append(word)
        gr["confs"].append(conf)
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        gr["x0"], gr["y0"] = min(gr["x0"], x), min(gr["y0"], y)
        gr["x1"], gr["y1"] = max(gr["x1"], x + w), max(gr["y1"], y + h)

    lines = []
    for gr in sorted(groups.values(), key=lambda g_: (g_["y0"], g_["x0"])):
        text = " ".join(gr["words"])
        conf = sum(gr["confs"]) / len(gr["confs"])
        is_time = bool(_TIME.match(text))
        keep = conf >= min_conf and not is_time and not _NOISE.match(text)
        lines.append({
            "text": text, "conf": round(conf, 1), "keep": keep,
            "reason": "heure" if is_time else ("confiance faible" if conf < min_conf else ""),
            "box": [round(v / scale) for v in (gr["x0"], gr["y0"], gr["x1"], gr["y1"])],
        })
    return {"lines": lines, "width": w0, "height": h0}


if __name__ == "__main__":
    res = ocr_lines(open(sys.argv[1], "rb").read())
    for l in res["lines"]:
        print(("✓ " if l["keep"] else "✗ ") + f"[{l['conf']:5.1f}] {l['text']}  {l['reason']}")
