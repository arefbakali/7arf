"""
7arf - Modèle encodeur-décodeur (PyTorch), écrit à la main.

  Entrée (caractères Arabizi)
    -> Embedding source
    -> Encodeur LSTM (uni- ou bidirectionnel)
    -> [Attention de Bahdanau]           (optionnelle : expérience A vs B)
    -> Décodeur LSTM (un caractère par pas de temps)
    -> Linear -> Softmax
  Sortie (caractères en écriture arabe)

Aucune couche Transformer, aucune auto-attention, aucun poids pré-entraîné.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .vocab import PAD_ID, SOS_ID, EOS_ID


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.n_dir = 2 if bidirectional else 1
        self.out_dim = hid_dim * self.n_dir

    def forward(self, src, src_len):
        """src : (B, S) ids ; src_len : (B,) longueurs réelles.
        Retour : outputs (B, S, out_dim) = h_1..h_S ; (h_n, c_n) de forme (L, B, out_dim)."""
        emb = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # (L*dir, B, H) -> (L, B, dir*H) : concaténation avant/arrière pour chaque couche
        B = src.size(0)
        h = h.view(self.n_layers, self.n_dir, B, -1).permute(0, 2, 1, 3).reshape(self.n_layers, B, -1)
        c = c.view(self.n_layers, self.n_dir, B, -1).permute(0, 2, 1, 3).reshape(self.n_layers, B, -1)
        return out, (h, c)


class BahdanauAttention(nn.Module):
    """Attention additive (Bahdanau, Cho & Bengio, 2015) entre décodeur et encodeur.

        e(t,i)  = v^T tanh(W_s s(t-1) + W_h h_i)
        a(t,i)  = softmax_i e(t,i)            (positions <PAD> masquées)
        c(t)    = sum_i a(t,i) h_i
    """

    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def precompute_keys(self, enc_out):
        return self.W_h(enc_out)  # (B, S, A), calculé une seule fois par séquence

    def forward(self, s_prev, keys, enc_out, mask):
        """s_prev : (B, Hd) ; keys : (B, S, A) ; enc_out : (B, S, E) ; mask : (B, S) True = vrai caractère."""
        scores = self.v(torch.tanh(keys + self.W_s(s_prev).unsqueeze(1))).squeeze(2)  # (B, S)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1)                                             # (B, S)
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)                  # (B, E)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, hid_dim, n_layers=1, use_attention=True,
                 dropout=0.3, attn_dim=256):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        rnn_in = emb_dim + (enc_dim if use_attention else 0)
        self.rnn = nn.LSTM(rnn_in, hid_dim, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        if use_attention:
            self.attention = BahdanauAttention(enc_dim, hid_dim, attn_dim)
            self.out = nn.Linear(hid_dim + enc_dim + emb_dim, vocab_size)
        else:
            self.out = nn.Linear(hid_dim, vocab_size)

    def step(self, y_prev, state, enc_out, keys, mask):
        """Un pas de décodage.
        y_prev : (B,) caractère précédent ; state : (h, c) chacun (L, B, Hd).
        Retour : logits (B, V), nouvel état, poids d'attention (B, S) ou None."""
        emb = self.dropout(self.embedding(y_prev))          # (B, E)
        alpha = None
        if self.use_attention:
            context, alpha = self.attention(state[0][-1], keys, enc_out, mask)
            rnn_in = torch.cat([emb, context], dim=1)
        else:
            rnn_in = emb
        out, state = self.rnn(rnn_in.unsqueeze(1), state)  # out : (B, 1, Hd)
        out = out.squeeze(1)
        if self.use_attention:
            logits = self.out(self.dropout(torch.cat([out, context, emb], dim=1)))
        else:
            logits = self.out(self.dropout(out))
        return logits, state, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=64, enc_hid=256, dec_hid=512, n_layers=1,
                 bidirectional=True, use_attention=True, dropout=0.3, attn_dim=256):
        super().__init__()
        self.config = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, emb_dim=emb_dim, enc_hid=enc_hid,
                           dec_hid=dec_hid, n_layers=n_layers, bidirectional=bidirectional,
                           use_attention=use_attention, dropout=dropout, attn_dim=attn_dim)
        self.encoder = Encoder(src_vocab, emb_dim, enc_hid, n_layers, bidirectional, dropout)
        self.decoder = Decoder(tgt_vocab, emb_dim, self.encoder.out_dim, dec_hid, n_layers,
                               use_attention, dropout, attn_dim)
        # "pont" : états finaux de l'encodeur -> état initial du décodeur s0
        self.bridge_h = nn.Linear(self.encoder.out_dim, dec_hid)
        self.bridge_c = nn.Linear(self.encoder.out_dim, dec_hid)

    def encode(self, src, src_len):
        enc_out, (h, c) = self.encoder(src, src_len)
        state = (torch.tanh(self.bridge_h(h)).contiguous(), torch.tanh(self.bridge_c(c)).contiguous())
        mask = src != PAD_ID
        keys = self.decoder.attention.precompute_keys(enc_out) if self.decoder.use_attention else None
        return enc_out, keys, mask, state

    def forward(self, src, src_len, tgt, teacher_forcing=1.0):
        """Entraînement. tgt : (B, T) = <SOS> y1 ... yn <EOS> <PAD>...
        Retour : logits (B, T-1, V) qui prédisent tgt[:, 1:]."""
        enc_out, keys, mask, state = self.encode(src, src_len)
        y = tgt[:, 0]
        logits_all = []
        for t in range(1, tgt.size(1)):
            logits, state, _ = self.decoder.step(y, state, enc_out, keys, mask)
            logits_all.append(logits)
            if teacher_forcing >= 1.0 or random.random() < teacher_forcing:
                y = tgt[:, t]                      # teacher forcing : vrai caractère précédent
            else:
                y = logits.argmax(1).detach()      # sinon : la propre prédiction du modèle
        return torch.stack(logits_all, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=None):
        """Décodage glouton par lot. Retour : liste d'ids par exemple, liste de matrices d'attention (T, S)."""
        enc_out, keys, mask, state = self.encode(src, src_len)
        B = src.size(0)
        max_len = max_len or int(src_len.max().item() * 2 + 10)
        y = torch.full((B,), SOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        outs, alphas = [], []
        for _ in range(max_len):
            logits, state, alpha = self.decoder.step(y, state, enc_out, keys, mask)
            y = logits.argmax(1)
            outs.append(y)
            if alpha is not None:
                alphas.append(alpha)
            done |= y == EOS_ID
            if done.all():  # critère d'arrêt : <EOS> pour tous, ou longueur maximale
                break
        outs = torch.stack(outs, 1).tolist()
        seqs, atts = [], []
        for b in range(B):
            seq = outs[b]
            n = seq.index(EOS_ID) if EOS_ID in seq else len(seq)
            seqs.append(seq[:n])
            if alphas:
                L = int(src_len[b])
                atts.append(torch.stack([a[b, :L] for a in alphas[:n + 1]]).cpu())
            else:
                atts.append(None)
        return seqs, atts

    @torch.no_grad()
    def beam_decode(self, src, src_len, beam_size=5, max_len=None, alpha_len=0.7):
        """Recherche en faisceau, un exemple à la fois (src : (1, S)).
        Score = log-probabilité / longueur^alpha_len (normalisation de longueur)."""
        assert src.size(0) == 1, "beam_decode traite un exemple à la fois"
        enc_out, keys, mask, (h, c) = self.encode(src, src_len)
        max_len = max_len or int(src_len.item() * 2 + 10)
        beams = [([SOS_ID], 0.0, (h, c))]           # (séquence, log-prob, état)
        finished = []
        for _ in range(max_len):
            k = len(beams)
            y = torch.tensor([b[0][-1] for b in beams], device=src.device)
            hs = torch.cat([b[2][0] for b in beams], dim=1)
            cs = torch.cat([b[2][1] for b in beams], dim=1)
            ek = enc_out.expand(k, -1, -1)
            kk = keys.expand(k, -1, -1) if keys is not None else None
            mk = mask.expand(k, -1)
            logits, (hn, cn), _ = self.decoder.step(y, (hs, cs), ek, kk, mk)
            logp = F.log_softmax(logits, dim=-1)       # (k, V)
            cand = []
            top_lp, top_id = logp.topk(beam_size, dim=-1)
            for i in range(k):
                for lp, tok in zip(top_lp[i].tolist(), top_id[i].tolist()):
                    cand.append((beams[i][1] + lp, i, tok))
            cand.sort(key=lambda x: x[0], reverse=True)
            new_beams = []
            for score, i, tok in cand:
                seq = beams[i][0] + [tok]
                st = (hn[:, i:i + 1].contiguous(), cn[:, i:i + 1].contiguous())
                if tok == EOS_ID:
                    finished.append((seq, score / (len(seq) - 1) ** alpha_len))
                else:
                    new_beams.append((seq, score, st))
                if len(new_beams) == beam_size:
                    break
            beams = new_beams
            if not beams or len(finished) >= beam_size:
                break
        if not finished:
            finished = [(b[0], b[1] / max(len(b[0]) - 1, 1) ** alpha_len) for b in beams]
        best = max(finished, key=lambda x: x[1])[0]
        return [t for t in best[1:] if t != EOS_ID]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
