
# ======================================================================
# 0.  Imports
# ======================================================================
import os, sys, re, gc, subprocess, math, torch, pandas as pd, numpy as np
from pathlib import Path
from typing import List
from wordfreq import word_frequency
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from utils import test_hypotheses_, create_plots

# ======================================================================
# 1.  Raw-data loader  (downloads the corpus if needed)
# ======================================================================
RAW_DATA_ROOT = Path("data/OneStop")

def load_raw_df() -> pd.DataFrame:
    """Return the IA-Paragraph CSV as a DataFrame (download if absent)."""
    # Clone repo if missing
    repo_dir = Path("OneStop-Eye-Movements")
    if not repo_dir.is_dir():
        subprocess.run(
            ["git", "clone",
             "https://github.com/lacclab/OneStop-Eye-Movements.git",
             str(repo_dir)],
            check=True,
        )

    # Run the download script if no matching CSV present
    has_csv = any(
        "ia_paragraph" in p.name.lower()
        for p in RAW_DATA_ROOT.rglob("*.csv")
    )
    if not has_csv:
        script = repo_dir / "download_data_files.py"
        subprocess.run(
            [sys.executable, str(script),
             "--mode", "ordinary",
             "--extract",
             "--output-folder", str(RAW_DATA_ROOT)],
            check=True,
        )

    # Locate and load the CSV (case-insensitive)
    for p in RAW_DATA_ROOT.rglob("*.csv"):
        if "ia_paragraph" in p.name.lower():
            return pd.read_csv(p)

    raise FileNotFoundError("Could not find ia_paragraph_ordinary.csv")

# ======================================================================
# 2.  Word-level surprisal & entropy helpers
# ======================================================================
# (unchanged)

def _compute_word_stats(text: str, tok, mdl, device) -> list[tuple[str, float, float]]:
    """
    Return [(word, surprisal_bits, entropy_bits), …] for `text`.
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids, offs = enc["input_ids"].to(device), enc["offset_mapping"][0]

    with torch.no_grad():
        logits = mdl(ids).logits
        lp = torch.log_softmax(logits, -1)

    surps, ents = [], []
    for i, tid in enumerate(ids[0][1:], start=1):
        row = lp[0, i - 1]
        surps.append((-row[tid].item()) / math.log(2))
        ents.append((-(row.exp() * row).sum().item()) / math.log(2))

    # map back to words
    spans, cur = [], 0
    for w in text.split():
        b = text.find(w, cur)
        spans.append((w, b, b + len(w)))
        cur = b + len(w)

    out, ptr, n = [], 0, len(surps)
    for w, s0, e0 in spans:
        s_acc = e_acc = 0.0
        while ptr < n:
            t0, t1 = offs[ptr + 1].tolist()
            if t0 >= e0:
                break
            if t1 > s0:
                s_acc += surps[ptr]
                e_acc += ents[ptr]
            ptr += 1
        out.append((w, s_acc, e_acc))
    return out


def add_surprisal_entropy_column(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    *,
    surprisal_col: str,
    entropy_col: str,
) -> pd.DataFrame:
    """Return new DF with two extra columns for the given model."""
    df = df.copy()

    # cache word-level stats per paragraph
    cache: dict[tuple, dict[str, tuple[float, float]]] = {}
    uniq = df[["article_batch", "article_id", "paragraph_id", "paragraph"]].drop_duplicates()

    for _, row in uniq.iterrows():
        key = (row.article_batch, row.article_id, row.paragraph_id)
        text = row.paragraph
        if isinstance(text, str) and text.strip():
            cache[key] = {w: (s, e)
                          for w, s, e in _compute_word_stats(text, tokenizer, model, device)}
        else:
            cache[key] = {}

    def _lookup(row, idx):
        return cache.get(
            (row.article_batch, row.article_id, row.paragraph_id), {}
        ).get(row.IA_LABEL, (np.nan, np.nan))[idx]

    df[surprisal_col] = df.apply(lambda r: _lookup(r, 0), axis=1)
    df[entropy_col]   = df.apply(lambda r: _lookup(r, 1), axis=1)
    return df

# ======================================================================
# 3.  Build-once / load-many cache utilities
# ======================================================================
ENRICHED_PARQUET = Path("data/OneStop/ia_paragraph_multi_enriched.parquet")
CACHE_DIR        = Path("data/OneStop/model_caches")

# (rest unchanged)
def _col_prefix(model_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", model_name.split("/")[-1]).replace("-", "_")

def _add_model_columns(df: pd.DataFrame,
                       model_name: str,
                       device: torch.device) -> pd.DataFrame:
    p           = _col_prefix(model_name)
    surp_col    = f"{p}_surprisal"
    ent_col     = f"{p}_entropy"
    cache_file  = CACHE_DIR / f"{p}.parquet"

    # already present?
    if surp_col in df.columns and ent_col in df.columns:
        return df

    # cached?
    if cache_file.is_file():
        cached = pd.read_parquet(cache_file)
        return pd.concat([df.reset_index(drop=True),
                          cached.reset_index(drop=True)], axis=1)

    # need to compute
    print(f"🧮  Computing {p} columns …")
    tok  = AutoTokenizer.from_pretrained(model_name)
    mdl  = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    df   = add_surprisal_entropy_column(df, tok, mdl, device,
                                        surprisal_col=surp_col, entropy_col=ent_col)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df[[surp_col, ent_col]].to_parquet(cache_file, index=False)
    print(f"✅  Cached columns → {cache_file}")
    del mdl; torch.cuda.empty_cache(); gc.collect()
    return df

def build_enriched_df(raw_df: pd.DataFrame,
                      model_list: List[str],
                      device: torch.device) -> pd.DataFrame:
    df = raw_df.copy()
    for m in model_list:
        df = _add_model_columns(df, m, device)
    ENRICHED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ENRICHED_PARQUET, index=False)
    print(f"💾  Saved full enriched DF → {ENRICHED_PARQUET}")
    return df

def load_enriched_df(raw_df_loader,
                     model_list: List[str],
                     device: torch.device) -> pd.DataFrame:
    if ENRICHED_PARQUET.is_file():
        df = pd.read_parquet(ENRICHED_PARQUET)
        # ensure it has all requested models
        for m in model_list:
            df = _add_model_columns(df, m, device)
        df.to_parquet(ENRICHED_PARQUET, index=False)
        return df

    # build from scratch
    raw_df = raw_df_loader()
    return build_enriched_df(raw_df, model_list, device)

# ======================================================================
# 5.  main()
# ======================================================================
def main():
    MODELS = [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_enriched_df(load_raw_df, MODELS, device)
    # Run hypothesis tests for each model
    for model_name in MODELS:
        prefix = _col_prefix(model_name)
        surp_col = f"{prefix}_surprisal"
        ent_col = f"{prefix}_entropy"
        print(f"\n### Results for {model_name} ###")
        test_hypotheses_(df, surp_col=surp_col, ent_col=ent_col,rt_col='IA_DWELL_TIME')

    for model_name in MODELS:
        prefix = _col_prefix(model_name)
        surp_col = f"{prefix}_surprisal"
        ent_col = f"{prefix}_entropy"
        create_plots(df, model_name, surp_col, ent_col, rt_col="IA_DWELL_TIME")

if __name__ == "__main__":
    main()
