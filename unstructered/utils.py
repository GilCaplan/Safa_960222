#!/usr/bin/env python3
"""
 test for surprisal + entropy analysis
Concise version with mini test first, then real data processing
"""

import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import torch
import pandas as pd
import numpy as np
from wordfreq import word_frequency
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
from scipy.stats import pearsonr

# ──────────────────────────────────────────────────────────────────────────────
# Low-level helper ─ word stats (surprisal + entropy)
# ──────────────────────────────────────────────────────────────────────────────
def compute_word_stats(
    text: str,
    tokenizer,
    model,
    device,
) -> list[tuple[str, float, float]]:
    """
    Returns a list of (word, surprisal_bits, entropy_bits) tuples.
      • surprisal = −log₂ p(word | context)
      • entropy   = Σ p(t)·−log₂ p(t)  of the *token distribution* that predicts
                    the next token(s) inside this word.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    offset_map = enc["offset_mapping"][0]        # (seq_len, 2)

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    # token-level surprisal & entropy for positions 1..N-1
    token_surps, token_entrs = [], []
    for i, tok in enumerate(input_ids[0][1:], start=1):
        lp_row = log_probs[0, i - 1]             # distribution that predicted tok
        token_surps.append(-lp_row[tok].item() / math.log(2))
        token_entrs.append(-(lp_row.exp() * lp_row).sum().item() / math.log(2))

    # word boundaries
    word_spans, cursor = [], 0
    for w in text.split():
        b = text.find(w, cursor)
        word_spans.append((w, b, b + len(w)))
        cursor = b + len(w)

    # aggregate token stats → word stats
    out, ptr, total = [], 0, len(token_surps)
    for word, w_start, w_end in word_spans:
        s_acc = e_acc = 0.0
        while ptr < total:
            t_start, t_end = offset_map[ptr + 1].tolist()
            if t_start >= w_end:
                break
            if t_end <= w_start:
                ptr += 1
                continue
            s_acc += token_surps[ptr]
            e_acc += token_entrs[ptr]
            ptr += 1
        out.append((word, s_acc, e_acc))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# High-level helper ─ add columns to a DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def add_surprisal_entropy_column(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    surprisal_col: str = "pythia_surprisal",
    entropy_col: str = "pythia_entropy",
) -> pd.DataFrame:
    """
    Returns a *new* DataFrame with:
      • word_length
      • log_frequency
      • surprisal in `surprisal_col`
      • entropy   in `entropy_col`
    Assumes columns: IA_LABEL, paragraph, article_batch, article_id, paragraph_id
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_out = df.copy()

    # easy lexical features
    df_out["word_length"] = df_out["IA_LABEL"].str.len()
    df_out["log_frequency"] = df_out["IA_LABEL"].apply(
        lambda w: np.log(word_frequency(str(w).lower(), "en") + 1e-8)
    )

    # cache word-level stats per paragraph
    cache: dict[tuple, dict[str, tuple[float, float]]] = {}
    uniq = df_out[["article_batch", "article_id", "paragraph_id", "paragraph"]].drop_duplicates()

    for _, row in uniq.iterrows():
        key = (row.article_batch, row.article_id, row.paragraph_id)
        para_text = row.paragraph
        if isinstance(para_text, str) and para_text.strip():
            cache[key] = {
                w: (s, e) for w, s, e in compute_word_stats(para_text, tokenizer, model, device)
            }
        else:
            cache[key] = {}

    def lookup(row, idx):
        key = (row.article_batch, row.article_id, row.paragraph_id)
        return cache.get(key, {}).get(row.IA_LABEL, (np.nan, np.nan))[idx]

    df_out[surprisal_col] = df_out.apply(lambda r: lookup(r, 0), axis=1)
    df_out[entropy_col] = df_out.apply(lambda r: lookup(r, 1), axis=1)
    return df_out

def test_hypotheses_(
    df,
    surp_col: str = "pythia_surprisal",
    ent_col:  str = "pythia_entropy",
    rt_col:   str = "reading_time",
):
    """
    Print R² for three regressions (surprisal, entropy, both)
    and indicate which models are statistically significant
    (overall F-test p-value < 0.05).
    """
    # --- checks --------------------------------------------------------------
    for col in (surp_col, ent_col, rt_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in DataFrame")

    df = df[[surp_col, ent_col, rt_col]].dropna()
    X_surp  = df[[surp_col]].values
    X_ent   = df[[ent_col]].values
    X_both  = df[[surp_col, ent_col]].values
    y       = df[rt_col].values

    # --- scikit-learn R² -----------------------------------------------------
    r2_surp = LinearRegression().fit(X_surp,  y).score(X_surp,  y)
    r2_ent  = LinearRegression().fit(X_ent,   y).score(X_ent,   y)
    r2_both = LinearRegression().fit(X_both,  y).score(X_both,  y)

    # --- statsmodels for overall p-values -----------------------------------
    sm_surp = sm.OLS(y, sm.add_constant(X_surp)).fit()
    sm_ent  = sm.OLS(y, sm.add_constant(X_ent)).fit()
    sm_both = sm.OLS(y, sm.add_constant(X_both)).fit()

    p_surp  = sm_surp.f_pvalue
    p_ent   = sm_ent.f_pvalue
    p_both  = sm_both.f_pvalue

    # --- output --------------------------------------------------------------
    print("\n=== MODEL COMPARISON ===")
    print(f"Surprisal only  : R² = {r2_surp:.4f} | significant: {'YES' if p_surp  < 0.05 else 'NO'} (p = {p_surp :.2e})")
    print(f"Entropy only    : R² = {r2_ent :.4f} | significant: {'YES' if p_ent   < 0.05 else 'NO'} (p = {p_ent  :.2e})")
    print(f"Combined model  : R² = {r2_both:.4f} | significant: {'YES' if p_both < 0.05 else 'NO'} (p = {p_both:.2e})")

    # optional: return stats as dict
    return {
        "r2_surp": r2_surp, "p_surp": p_surp,
        "r2_ent":  r2_ent,  "p_ent":  p_ent,
        "r2_both": r2_both, "p_both": p_both,
    }


import matplotlib.pyplot as plt

def create_plots(
    df: pd.DataFrame,
    model_name: str,
    surp_col: str,
    ent_col: str,
    rt_col:  str = "IA_DWELL_TIME",
):
    """
    Display two scatterplots with regression lines:
      1) Reading Time vs. Surprisal
      2) Reading Time vs. Entropy
    """

    if len(df) < 10:
        print(f"{model_name}: Not enough data for plotting")
        return

    # Helper to fit & plot
    def _scatter_with_line(x, y, xlabel, ylabel, title):
        # drop NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_, y_ = x[mask], y[mask]

        # fit regression
        lr = LinearRegression().fit(x_.reshape(-1, 1), y_)
        r2 = lr.score(x_.reshape(-1, 1), y_)
        # for a smooth line, sort by x
        sort_idx = np.argsort(x_)
        xs = x_[sort_idx].reshape(-1, 1)
        ys_pred = lr.predict(xs)

        plt.figure(figsize=(6, 4))
        plt.scatter(x_, y_, alpha=0.5, s=10)
        plt.plot(xs, ys_pred, linewidth=2,color="r")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{model_name} — {title}\n$R^2$ = {r2:.3f}")
        plt.tight_layout()
        plt.show()

    # 1) RT vs Surprisal
    _scatter_with_line(
        df[surp_col].values,
        df[rt_col].values,
        xlabel="Surprisal (bits)",
        ylabel="Reading Time (ms)",
        title="RT vs Surprisal"
    )

    # 2) RT vs Entropy
    _scatter_with_line(
        df[ent_col].values,
        df[rt_col].values,
        xlabel="Entropy (bits)",
        ylabel="Reading Time (ms)",
        title="RT vs Entropy"
    )

    _scatter_with_line(
        df[ent_col].values + df[surp_col].values,
        df[rt_col].values,
        xlabel="Surprisal + Entropy (bits)",
        ylabel="Reading Time (ms)",
        title="RT vs Surprisal + Entropy"
    )