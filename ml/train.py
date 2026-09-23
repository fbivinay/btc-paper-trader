"""LSTM training with walk-forward validation across candidate training windows.

Answers one question: how much history should the production model be trained on?
Each candidate window is trained and tested on several rolling out-of-sample
periods, and scored on trading metrics rather than accuracy.

Run locally first to check the pipeline:
    python ml/train.py --smoke
Then the full sweep (slow on CPU, meant for Kaggle GPU):
    python ml/train.py --windows 180 365 730 --folds 4
"""

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from metrics import evaluate, DOWN, NEUTRAL, UP

DATA = Path(os.environ.get("BTC_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
ART = Path(os.environ.get("BTC_ARTIFACT_DIR",
                          Path(__file__).resolve().parent.parent / "artifacts"))
SRC = DATA / "BTCUSDT_5m_features.parquet"

SEQ_LEN = 120          # 120 x 5m = 10 hours of context
BARS_PER_DAY = 288
BASE = "top100pct"          # the ungated gate name, used as the headline metric
NOT_FEATURES = {"open_time", "open", "high", "low", "close", "volume",
                "is_gap", "fwd_return", "label"}


class SeqDataset(Dataset):
    """Yields the SEQ_LEN bars ending at each index.

    Sequences are sliced on demand from one contiguous array. Materialising them
    would cost 120x the memory for data that is 99% overlap.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray):
        self.x, self.y, self.idx = x, y, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = self.idx[k]
        return self.x[i - SEQ_LEN + 1: i + 1], self.y[i]


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])          # last timestep only


def folds(df: pd.DataFrame, train_days: int, test_days: int, n_folds: int):
    """Rolling out-of-sample splits, newest fold last. Train always precedes test."""
    t = df["open_time"]
    end = t.iloc[-1]
    out = []
    for k in reversed(range(n_folds)):
        test_hi = end - pd.Timedelta(days=k * test_days)
        test_lo = test_hi - pd.Timedelta(days=test_days)
        train_lo = test_lo - pd.Timedelta(days=train_days)
        if train_lo < t.iloc[0]:
            continue
        tr = np.flatnonzero((t >= train_lo) & (t < test_lo))
        te = np.flatnonzero((t >= test_lo) & (t < test_hi))
        if len(tr) and len(te):
            out.append((tr, te, test_lo.date(), test_hi.date()))
    return out


def run_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(xb)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = [torch.softmax(model(xb.to(device)), dim=1).cpu() for xb, _ in loader]
    p = torch.cat(probs).numpy()
    return p.argmax(1), p.max(1)


def train_fold(df, feat_cols, tr_idx, te_idx, args, device):
    x_all = df[feat_cols].to_numpy(np.float32)
    y_all = df["label"].to_numpy(np.int64)
    fwd_all = df["fwd_return"].to_numpy(np.float64)

    # Scaler fitted on the training window only. Fitting it over the whole file
    # would leak the test period's mean and variance into training.
    mu = x_all[tr_idx].mean(0)
    sd = x_all[tr_idx].std(0)
    sd[sd < 1e-8] = 1.0
    x = (x_all - mu) / sd
    np.clip(x, -10, 10, out=x)

    # Only bars with a full SEQ_LEN of history behind them, and no gap-filled bars.
    gap = df["is_gap"].to_numpy()
    def usable(idx, stride):
        idx = idx[idx >= SEQ_LEN - 1]
        idx = idx[~gap[idx]]
        return idx[::stride]

    tr = usable(tr_idx, args.stride)
    val_cut = int(len(tr) * 0.9)
    tr, val = tr[:val_cut], tr[val_cut:]
    te = usable(te_idx, 1)
    if not len(tr) or not len(te):
        return None

    mk = lambda idx, shuffle: DataLoader(
        SeqDataset(x, y_all, idx), batch_size=args.batch, shuffle=shuffle, num_workers=0)
    tr_dl, val_dl, te_dl = mk(tr, True), mk(val, False), mk(te, False)

    # NEUTRAL is ~68% of labels; without reweighting the model predicts it forever.
    counts = np.bincount(y_all[tr], minlength=3).astype(np.float64)
    weights = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))), dtype=torch.float32)

    model = LSTMClassifier(len(feat_cols), args.hidden, args.layers, args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_state, patience = -np.inf, None, 0
    for ep in range(args.epochs):
        loss = run_epoch(model, tr_dl, loss_fn, opt, device)
        vp, vc = predict(model, val_dl, device)
        # Selected on validation trading return, not loss. Loss rewards being
        # confidently NEUTRAL, which earns nothing.
        score = evaluate(vp, vc, fwd_all[val], y_all[val])["total_return"]
        flag = ""
        if score > best:
            best, best_state, patience, flag = score, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, 0, "  *"
        else:
            patience += 1
        print(f"    epoch {ep + 1:>2}  loss {loss:.4f}  val_return {score:+.2%}{flag}", flush=True)
        if patience >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Confidence gates come from the VALIDATION distribution, not fixed absolute
    # values. A weighted 3-class softmax rarely exceeds 0.5, so thresholds like
    # 0.7 silently select zero trades. Quantiles read as "trade only the most
    # confident N% of signals" and stay leak-free because val precedes test.
    vp, vc = predict(model, val_dl, device)
    directional = vc[vp != NEUTRAL]
    if len(directional):
        gates = {f"top{int((1 - q) * 100)}pct": float(np.quantile(directional, q))
                 for q in (0.0, 0.5, 0.75, 0.9)}
    else:
        gates = {"top100pct": 0.0}

    tp, tc = predict(model, te_dl, device)
    res = {name: evaluate(tp, tc, fwd_all[te], y_all[te], min_conf=g)
           for name, g in gates.items()}
    return {"metrics": res, "state": model.state_dict(), "mu": mu, "sd": sd,
            "n_train": len(tr), "n_test": len(te), "gates": gates,
            "conf_mean": float(tc.mean()), "conf_max": float(tc.max()),
            "pred_mix": {n: float((tp == c).mean()) for n, c in
                         (("down", DOWN), ("neutral", NEUTRAL), ("up", UP))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[180, 365, 730])
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    # Consecutive bars share 119 of 120 timesteps. Striding cuts epoch cost with
    # almost no information loss.
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="one short fold on one window, to prove the pipeline runs")
    args = ap.parse_args()

    if args.smoke:
        args.windows, args.folds, args.epochs, args.stride = [180], 1, 2, 12
        args.test_days, args.patience = 30, 99

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(SRC)
    feat_cols = [c for c in df.columns if c not in NOT_FEATURES]
    print(f"{len(df):,} rows  {len(feat_cols)} features  device={device}")

    ART.mkdir(exist_ok=True)
    summary = []

    for w in args.windows:
        fs = folds(df, w, args.test_days, args.folds)
        if not fs:
            print(f"\nwindow {w}d: not enough history, skipped")
            continue
        print(f"\n=== training window {w}d, {len(fs)} folds ===")

        per_fold = []
        for tr_idx, te_idx, lo, hi in fs:
            print(f"  fold test {lo} .. {hi}  ({len(tr_idx):,} train bars)", flush=True)
            t0 = time.time()
            out = train_fold(df, feat_cols, tr_idx, te_idx, args, device)
            if out is None:
                print("    skipped, empty split")
                continue
            m = out["metrics"][BASE]
            mix = out["pred_mix"]
            print(f"    test: trades {m['n_trades']:>4}  win {m['win_rate']:.1%}  "
                  f"ret {m['total_return']:+.2%}  sharpe {m['sharpe']:+.2f}  "
                  f"dd {m['max_dd']:.2%}  prec_up {m['prec_up']:.1%}  "
                  f"prec_dn {m['prec_down']:.1%}  [{time.time() - t0:.0f}s]", flush=True)
            print(f"    mix: up {mix['up']:.1%} / neutral {mix['neutral']:.1%} / "
                  f"down {mix['down']:.1%}   conf mean {out['conf_mean']:.3f} "
                  f"max {out['conf_max']:.3f}", flush=True)
            for name, g in out["metrics"].items():
                if name != BASE:
                    print(f"      gate {name:>10} (conf>={out['gates'][name]:.3f}): "
                          f"trades {g['n_trades']:>4}  ret {g['total_return']:+.2%}  "
                          f"win {g['win_rate']:.1%}", flush=True)
            per_fold.append({"test_from": str(lo), "test_to": str(hi),
                             "pred_mix": out["pred_mix"], "gates": out["gates"],
                             **out["metrics"]})
            torch.save({"state": out["state"], "mu": out["mu"], "sd": out["sd"],
                        "feat_cols": feat_cols, "seq_len": SEQ_LEN,
                        "window_days": w, "test_to": str(hi),
                        "hidden": args.hidden, "layers": args.layers},
                       ART / f"model_w{w}_{hi}.pt")

        if per_fold:
            agg = {k: float(np.mean([f[BASE][k] for f in per_fold]))
                   for k in ("total_return", "sharpe", "win_rate", "max_dd",
                             "profit_factor", "prec_up", "prec_down", "n_trades")}
            # Consistency across folds matters more than the mean: a window that
            # only works in one regime is not a window worth deploying.
            agg["return_std"] = float(np.std([f[BASE]["total_return"] for f in per_fold]))
            agg["folds_profitable"] = int(sum(f[BASE]["total_return"] > 0 for f in per_fold))
            # A model that only ever predicts one direction is not predicting.
            agg["dir_balance"] = float(np.mean([
                min(f["pred_mix"]["up"], f["pred_mix"]["down"])
                / max(f["pred_mix"]["up"], f["pred_mix"]["down"], 1e-9) for f in per_fold]))
            summary.append({"window_days": w, "n_folds": len(per_fold),
                            "mean": agg, "folds": per_fold})
            print(f"  MEAN  ret {agg['total_return']:+.2%} (sd {agg['return_std']:.2%})  "
                  f"sharpe {agg['sharpe']:+.2f}  profitable folds {agg['folds_profitable']}/{len(per_fold)}")

    (ART / "walkforward.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ART / 'walkforward.json'}")

    if summary:
        print(f"\n{'window':>8} {'folds':>6} {'mean ret':>10} {'sd':>8} {'sharpe':>8} "
              f"{'prec_up':>8} {'prec_dn':>8} {'profitable':>11}")
        for s in summary:
            m = s["mean"]
            print(f"{s['window_days']:>7}d {s['n_folds']:>6} {m['total_return']:>+10.2%} "
                  f"{m['return_std']:>8.2%} {m['sharpe']:>+8.2f} {m['prec_up']:>8.1%} "
                  f"{m['prec_down']:>8.1%} {m['dir_balance']:>8.2f} "
                  f"{m['folds_profitable']:>7}/{s['n_folds']}")


if __name__ == "__main__":
    main()
