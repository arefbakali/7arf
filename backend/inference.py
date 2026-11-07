"""7arf - Inférence : chargement d'un checkpoint et translittération (glouton / faisceau)."""
import os

import torch

from .model import Seq2Seq
from .vocab import CharVocab, SOS_ID


def load_model(run_dir, device="cpu"):
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    model = Seq2Seq(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    src_vocab = CharVocab.load(os.path.join(run_dir, "src_vocab.json"))
    tgt_vocab = CharVocab.load(os.path.join(run_dir, "tgt_vocab.json"))
    # Les checkpoints du notebook contiennent {state_dict, config, epoch, val_cer} (pas de clé "args").
    return model, src_vocab, tgt_vocab, ck.get("args", ck.get("config"))


def transliterate(model, src_vocab, tgt_vocab, text, device="cpu", beam=0, max_src_len=120):
    """Translittère une chaîne Arabizi déjà normalisée. Retour : (sortie, matrice d'attention ou None)."""
    ids = src_vocab.encode(text, max_len=max_src_len) or [src_vocab.stoi.get(" ", 1)]
    src = torch.tensor([ids], device=device)
    src_len = torch.tensor([len(ids)])
    if beam and beam > 1:
        out = model.beam_decode(src, src_len, beam_size=beam)
        # recalcul de l'attention le long de la sortie choisie (pour la visualiser)
        att = attention_for_output(model, src, src_len, out) if model.decoder.use_attention else None
        return tgt_vocab.decode(out), att
    seqs, atts = model.greedy_decode(src, src_len)
    return tgt_vocab.decode(seqs[0]), atts[0]


@torch.no_grad()
def attention_for_output(model, src, src_len, out_ids):
    enc_out, keys, mask, state = model.encode(src, src_len)
    y = torch.tensor([SOS_ID], device=src.device)
    rows = []
    for tok in out_ids:
        _, state, a = model.decoder.step(y, state, enc_out, keys, mask)
        rows.append(a[0, :int(src_len[0])].cpu())
        y = torch.tensor([tok], device=src.device)
    return torch.stack(rows) if rows else None
