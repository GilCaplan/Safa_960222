import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from tqdm import tqdm
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
import os

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data():
    """Load OneStop dataset with proper error handling - FIXED paths"""
    print("Loading OneStop dataset...")

    # Try multiple possible paths
    possible_paths = [
        "ia_Paragraph.csv",  # Current directory
        "data/onestop/ia_Paragraph.csv",  # Data subfolder
        "../data/onestop/ia_Paragraph.csv",  # Parent data folder
        "onestop/ia_Paragraph.csv",  # Direct onestop folder
        Path.cwd() / "ia_Paragraph.csv"  # Explicit current directory
    ]

    raw_data = None
    for file_path in possible_paths:
        try:
            file_path = Path(file_path)
            if file_path.exists():
                raw_data = pd.read_csv(file_path)
                print(f"✓ Found data at: {file_path}")
                break
        except Exception as e:
            continue

    if raw_data is None:
        raise FileNotFoundError(
            f"Could not find ia_Paragraph.csv in any of these locations:\n" +
            "\n".join([f"  - {p}" for p in possible_paths]) +
            "\nPlease ensure the file is in your current directory or update the path."
        )

    print(f"Loaded {len(raw_data)} rows")
    print(f"Shape: {raw_data.shape}")

    # Check for required columns (from Task 1)
    required_cols = ['participant_id', 'TRIAL_INDEX', 'IA_LABEL', 'IA_DWELL_TIME', 'IA_ID']
    missing_cols = [col for col in required_cols if col not in raw_data.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print("Available columns:")
        for col in raw_data.columns[:10]:  # Show first 10
            print(f"  - {col}")
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("✓ All required columns found")
    return raw_data


def load_pythia_70m():
    """Load Pythia 70M model and tokenizer with error handling"""
    print("Loading Pythia 70M model...")
    model_name = "EleutherAI/pythia-70m"

    # Device selection with proper fallback
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.float16
    else:
        device = torch.device('cpu')
        dtype = torch.float32
        print("⚠️  Using CPU - this will be slower")

    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=None)
        model.to(device).eval()

        print(f"✓ Model loaded successfully on {device}")
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have internet connection and sufficient disk space")
        raise


def compute_word_surprisal(context, target_word, model, tokenizer, device):
    """
    Compute surprisal of target_word given context - EXACTLY like Task 1
    """
    try:
        # Construct full text
        if context.strip():
            full_text = context + " " + target_word
            context_text = context
        else:
            full_text = target_word
            context_text = ""

        # Tokenize full text with offset mapping
        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True
        )

        input_ids = encoding["input_ids"].to(device)
        offset_mapping = encoding["offset_mapping"][0]

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

        # Find tokens that correspond to the target word
        target_start = len(context_text)
        if context_text:
            target_start += 1  # Account for space
        target_end = target_start + len(target_word)

        # Find tokens for target word
        target_token_surprisals = []
        for i in range(1, input_ids.size(1)):  # Skip BOS token
            token_start, token_end = offset_mapping[i].tolist()

            # Check if this token overlaps with target word
            if token_start < target_end and token_end > target_start:
                token_id = input_ids[0, i].item()
                token_log_prob = log_probs[0, i - 1, token_id].item()
                surprisal = -token_log_prob / math.log(10)  # Base 10 like Task 1
                target_token_surprisals.append(surprisal)

        # Sum surprisals for target word (like Task 1)
        if target_token_surprisals:
            word_surprisal = sum(target_token_surprisals)
        else:
            word_surprisal = 5.0  # Reasonable fallback for errors

        return word_surprisal  # NO CLAMPING - keep natural values

    except Exception as e:
        return 5.0  # Reasonable fallback on error


def compute_entropy_at_position(context, model, tokenizer, device):
    """
    Compute entropy of model's prediction given context
    """
    try:
        if not context.strip():
            return 6.0  # Reasonable default for empty context

        # Tokenize context
        encoding = tokenizer(context, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding["input_ids"].to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last position predictions

        # Calculate entropy using top-k for efficiency
        top_k_logits, _ = torch.topk(logits, min(1000, logits.size(0)))
        top_k_probs = F.softmax(top_k_logits, dim=0)

        # Entropy in base 10
        entropy = -(top_k_probs * torch.log10(top_k_probs + 1e-10)).sum().item()

        return entropy  # NO CLAMPING - keep natural values

    except Exception as e:
        return 6.0  # Reasonable fallback


def process_dataset(df, model, tokenizer, device, max_trials=None):
    """
    Process dataset word-by-word like Task 1 - NO BATCHING
    """
    print(f"Starting word-by-word processing...")
    print(f"Input data shape: {df.shape}")

    # EXACT Task 1 preprocessing
    word_col = 'IA_LABEL'
    rt_col = 'IA_DWELL_TIME'
    participant_col = 'participant_id'
    trial_col = 'TRIAL_INDEX'

    # Basic preprocessing like Task 1
    df = df.dropna(subset=[word_col, rt_col, 'IA_ID'])
    df = df[df[rt_col] > 0]
    df = df[df[word_col].str.len() > 0]
    df = df[df[word_col].str.isalpha()]  # Only alphabetic words

    # Sort by reading order using IA_ID
    df = df.sort_values([participant_col, trial_col, 'IA_ID'])

    # Remove outliers (EXACT Task 1 approach)
    q99 = df[rt_col].quantile(0.99)
    q01 = df[rt_col].quantile(0.01)
    df = df[(df[rt_col] >= q01) & (df[rt_col] <= q99)]

    print(f"After preprocessing: {len(df)} words from {df[participant_col].nunique()} participants")

    # Group by participant and trial
    trials = list(df.groupby([participant_col, trial_col]))
    total_available = len(trials)

    # Limit trials if specified (for testing)
    if max_trials is not None:
        if len(trials) > max_trials:
            trials = trials[:max_trials]
            print(f"Processing {max_trials}/{total_available} trials for testing")
        else:
            print(f"Processing ALL {len(trials)} trials (less than limit)")
    else:
        print(f"Processing ALL {len(trials)} trials (no limit set)")

    results = []
    processed_trials = 0
    skipped_trials = 0

    # Process each trial (paragraph) word by word
    for (participant, trial_idx), group in tqdm(trials, desc="Processing trials"):
        processed_trials += 1

        # Sort by IA_ID to get correct word order
        group = group.sort_values('IA_ID').reset_index(drop=True)

        # Filter trial length like Task 1
        if len(group) < 3 or len(group) > 100:
            skipped_trials += 1
            continue

        # Process each word in the trial
        words = group[word_col].tolist()
        reading_times = group[rt_col].tolist()

        # Build context incrementally and process each word
        for word_idx, (word, rt) in enumerate(zip(words, reading_times)):
            try:
                # Build context from preceding words
                context_words = words[:word_idx]  # All preceding words
                context = ' '.join(context_words) if context_words else ""

                # Compute surprisal of current word given context
                surprisal = compute_word_surprisal(context, word, model, tokenizer, device)

                # Compute entropy at this position
                entropy = compute_entropy_at_position(context, model, tokenizer, device)

                # Store result (no artificial clamping)
                results.append({
                    'participant_id': participant,
                    'trial_index': trial_idx,
                    'word_position': word_idx + 1,
                    'word': word,
                    'reading_time': rt,
                    'pythia_surprisal': surprisal,  # Natural values
                    'pythia_entropy': entropy,  # Natural values
                    'sentence_length': len(words),
                    'context_length': len(context_words)
                })

            except Exception as e:
                continue  # Skip problematic words

    result_df = pd.DataFrame(results)
    print(f"✓ Processed {len(result_df)} words from {processed_trials} trials")
    print(f"  Skipped {skipped_trials} trials due to length filtering")
    if processed_trials > 0:
        print(f"  Average words per trial: {len(result_df) / processed_trials:.1f}")

    return result_df


def quick_test_model(model, tokenizer, device):
    """Test model with FIXED word-by-word calculation"""
    print("\n" + "=" * 50)
    print("TESTING PYTHIA MODEL WITH WORD-BY-WORD CALCULATION")
    print("=" * 50)

    test_cases = [
        (["The", "cat", "sat", "on", "the", "mat"]),
        (["This", "is", "a", "simple", "test"]),
        (["Machine", "learning", "models", "predict", "text"])
    ]

    for i, words in enumerate(test_cases, 1):
        print(f"\nTest {i}: {' '.join(words)}")

        surprisals = []
        entropies = []

        # Process each word with preceding context
        for word_idx, word in enumerate(words):
            context_words = words[:word_idx]
            context = ' '.join(context_words) if context_words else ""

            surprisal = compute_word_surprisal(context, word, model, tokenizer, device)
            entropy = compute_entropy_at_position(context, model, tokenizer, device)

            surprisals.append(surprisal)
            entropies.append(entropy)

        print(f"Words: {words}")
        print(f"Surprisals: {[f'{s:.2f}' for s in surprisals]}")
        print(f"Entropies: {[f'{e:.2f}' for e in entropies]}")

        # Check for variation
        if len(set([round(s, 1) for s in surprisals])) > 1:
            print("✓ Surprisal calculation shows variation")
        else:
            print("❌ Surprisals all too similar")

    print("\n" + "=" * 50)
    print("MODEL TEST COMPLETE")
    print("=" * 50)


def validate_results(results_df):
    """Validate the calculated surprisal and entropy values"""
    print("\n" + "=" * 50)
    print("VALIDATING RESULTS")
    print("=" * 50)

    if len(results_df) == 0:
        print("❌ No data to validate")
        return False

    # Check for required columns
    required_cols = ['word', 'pythia_surprisal', 'pythia_entropy', 'reading_time']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False

    # Check data quality
    surprisal_valid = results_df['pythia_surprisal'].dropna()
    entropy_valid = results_df['pythia_entropy'].dropna()

    print(f"Data coverage:")
    print(f"  Total observations: {len(results_df)}")
    print(f"  Valid surprisal: {len(surprisal_valid)} ({len(surprisal_valid) / len(results_df) * 100:.1f}%)")
    print(f"  Valid entropy: {len(entropy_valid)} ({len(entropy_valid) / len(results_df) * 100:.1f}%)")

    # Check value ranges and variation
    if len(surprisal_valid) > 0:
        surprisal_range_ok = (surprisal_valid.min() >= 0) and (surprisal_valid.max() <= 100.0)  # Natural range
        surprisal_variation = surprisal_valid.std()
        print(
            f"  Surprisal range OK: {'✓' if surprisal_range_ok else '❌'} ({surprisal_valid.min():.2f} - {surprisal_valid.max():.2f})")
        print(f"  Surprisal variation: {surprisal_variation:.2f} (should be > 1.0)")

        # Check for realistic values
        realistic_values = (surprisal_valid > 0.5).sum() / len(surprisal_valid)
        print(f"  Realistic surprisal values: {realistic_values:.1%} (should be > 50%)")
    else:
        surprisal_range_ok = False
        surprisal_variation = 0

    if len(entropy_valid) > 0:
        entropy_range_ok = (entropy_valid.min() >= 0) and (entropy_valid.max() <= 50.0)  # Natural range
        print(
            f"  Entropy range OK: {'✓' if entropy_range_ok else '❌'} ({entropy_valid.min():.2f} - {entropy_valid.max():.2f})")
    else:
        entropy_range_ok = False

    # Check minimum viable dataset size
    min_size_ok = len(results_df) >= 1000
    print(f"  Minimum dataset size: {'✓' if min_size_ok else '❌'} ({len(results_df)} observations)")

    # Overall validation
    coverage_ok = (len(surprisal_valid) >= len(results_df) * 0.8 and
                   len(entropy_valid) >= len(results_df) * 0.8)
    variation_ok = surprisal_variation > 1.0

    is_valid = (coverage_ok and surprisal_range_ok and entropy_range_ok and
                min_size_ok and variation_ok)

    print(f"\nOverall validation: {'✓ PASSED' if is_valid else '❌ FAILED'}")

    if not is_valid:
        print("Issues detected:")
        if not coverage_ok:
            print("  - Low data coverage")
        if not variation_ok:
            print("  - Insufficient surprisal variation")
        if not min_size_ok:
            print("  - Dataset too small")
        print("⚠️  Continuing with available data...")
        return len(surprisal_valid) > 500 and variation_ok

    return is_valid


def fit_regression_models(df):
    """
    Fit three regression models to compare predictors of reading time
    """
    print("\n=== FITTING REGRESSION MODELS ===")

    # Clean data
    clean_df = df.dropna(subset=['pythia_surprisal', 'pythia_entropy', 'reading_time'])

    if len(clean_df) < 1000:
        print(f"❌ Insufficient data for modeling: {len(clean_df)} observations")
        return None, clean_df

    print(f"Using {len(clean_df)} clean observations for modeling")

    X_surprisal = clean_df[['pythia_surprisal']].values
    X_entropy = clean_df[['pythia_entropy']].values
    X_combined = clean_df[['pythia_surprisal', 'pythia_entropy']].values
    y = clean_df['reading_time'].values

    models = {}

    try:
        # 1. Surprisal only model
        print("\n1. Surprisal-only model:")
        model_surprisal = LinearRegression().fit(X_surprisal, y)
        y_pred_surprisal = model_surprisal.predict(X_surprisal)
        r2_surprisal = r2_score(y, y_pred_surprisal)

        X_surprisal_sm = sm.add_constant(X_surprisal)
        sm_model_surprisal = sm.OLS(y, X_surprisal_sm).fit()

        print(f"  R² = {r2_surprisal:.4f}")
        print(f"  Coefficient = {model_surprisal.coef_[0]:.3f}")
        print(f"  p-value = {sm_model_surprisal.pvalues[1]:.6f}")
        print(f"  AIC = {sm_model_surprisal.aic:.2f}")

        models['surprisal'] = {
            'model': model_surprisal,
            'r2': r2_surprisal,
            'aic': sm_model_surprisal.aic,
            'bic': sm_model_surprisal.bic,
            'coefficients': model_surprisal.coef_,
            'pvalues': sm_model_surprisal.pvalues[1:],
            'sm_model': sm_model_surprisal
        }

        # 2. Entropy only model
        print("\n2. Entropy-only model:")
        model_entropy = LinearRegression().fit(X_entropy, y)
        y_pred_entropy = model_entropy.predict(X_entropy)
        r2_entropy = r2_score(y, y_pred_entropy)

        X_entropy_sm = sm.add_constant(X_entropy)
        sm_model_entropy = sm.OLS(y, X_entropy_sm).fit()

        print(f"  R² = {r2_entropy:.4f}")
        print(f"  Coefficient = {model_entropy.coef_[0]:.3f}")
        print(f"  p-value = {sm_model_entropy.pvalues[1]:.6f}")
        print(f"  AIC = {sm_model_entropy.aic:.2f}")

        models['entropy'] = {
            'model': model_entropy,
            'r2': r2_entropy,
            'aic': sm_model_entropy.aic,
            'bic': sm_model_entropy.bic,
            'coefficients': model_entropy.coef_,
            'pvalues': sm_model_entropy.pvalues[1:],
            'sm_model': sm_model_entropy
        }

        # 3. Combined model
        print("\n3. Combined model (Surprisal + Entropy):")
        model_combined = LinearRegression().fit(X_combined, y)
        y_pred_combined = model_combined.predict(X_combined)
        r2_combined = r2_score(y, y_pred_combined)

        X_combined_sm = sm.add_constant(X_combined)
        sm_model_combined = sm.OLS(y, X_combined_sm).fit()

        print(f"  R² = {r2_combined:.4f}")
        print(f"  Surprisal coeff = {model_combined.coef_[0]:.3f}")
        print(f"  Entropy coeff = {model_combined.coef_[1]:.3f}")
        print(f"  Surprisal p-value = {sm_model_combined.pvalues[1]:.6f}")
        print(f"  Entropy p-value = {sm_model_combined.pvalues[2]:.6f}")
        print(f"  AIC = {sm_model_combined.aic:.2f}")

        models['combined'] = {
            'model': model_combined,
            'r2': r2_combined,
            'aic': sm_model_combined.aic,
            'bic': sm_model_combined.bic,
            'coefficients': model_combined.coef_,
            'pvalues': sm_model_combined.pvalues[1:],
            'sm_model': sm_model_combined
        }

        # Compare to Task 1 expectations
        print(f"\n=== COMPARISON TO TASK 1 ===")
        print(f"Expected surprisal R² (from Task 1): ~0.026")
        print(f"Current surprisal R²: {r2_surprisal:.4f}")
        if r2_surprisal > 0.001:
            print(f"Relative strength: {r2_surprisal / 0.026 * 100:.1f}% of Task 1 effect")
        else:
            print("R² too low for meaningful comparison")

        return models, clean_df

    except Exception as e:
        print(f"❌ Error fitting models: {e}")
        import traceback
        traceback.print_exc()
        return None, clean_df


def compare_models(models):
    """Compare the three models using statistical tests and information criteria"""
    print("\n=== MODEL COMPARISON ===")

    if not models:
        print("❌ No models to compare")
        return None

    # Extract performance metrics
    results = []
    for name, model_info in models.items():
        results.append({
            'Model': name.title(),
            'R²': model_info['r2'],
            'AIC': model_info['aic'],
            'BIC': model_info['bic'],
            'Coefficients': len(model_info['coefficients'])
        })

    comparison_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))

    # Determine best model by different criteria
    best_r2 = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']

    print(f"\nBest models:")
    print(f"  Highest R²: {best_r2} (R² = {comparison_df['R²'].max():.4f})")
    print(f"  Lowest AIC: {best_aic} (AIC = {comparison_df['AIC'].min():.2f})")
    print(f"  Lowest BIC: {best_bic} (BIC = {comparison_df['BIC'].min():.2f})")

    # Calculate R² improvements
    r2_surprisal = models['surprisal']['r2']
    r2_entropy = models['entropy']['r2']
    r2_combined = models['combined']['r2']

    print(f"\nR² Improvements:")
    if r2_surprisal > 0:
        improvement_entropy = ((r2_entropy - r2_surprisal) / r2_surprisal) * 100
        improvement_combined = ((r2_combined - r2_surprisal) / r2_surprisal) * 100
    else:
        improvement_entropy = 0
        improvement_combined = 0

    print(f"  Entropy vs Surprisal: {improvement_entropy:+.1f}%")
    print(f"  Combined vs Surprisal: {improvement_combined:+.1f}%")

    print(f"\nAdditional R² from entropy: {r2_combined - r2_surprisal:.4f}")

    return comparison_df


def test_hypotheses(models, df):
    """Test specific hypotheses about entropy and surprisal effects"""
    print("\n=== HYPOTHESIS TESTING ===")

    if not models:
        print("❌ No models for hypothesis testing")
        return

    surprisal_model = models['surprisal']['sm_model']
    entropy_model = models['entropy']['sm_model']
    combined_model = models['combined']['sm_model']

    print("\nHypothesis 1: Surprisal significantly predicts reading time")
    surprisal_significant = surprisal_model.pvalues[1] < 0.05
    print(f"  Result: {'✓ SUPPORTED' if surprisal_significant else '❌ NOT SUPPORTED'}")
    print(f"  p-value: {surprisal_model.pvalues[1]:.6f}")
    print(f"  Effect size (β): {surprisal_model.params[1]:.3f}")

    print("\nHypothesis 2: Entropy significantly predicts reading time")
    entropy_significant = entropy_model.pvalues[1] < 0.05
    print(f"  Result: {'✓ SUPPORTED' if entropy_significant else '❌ NOT SUPPORTED'}")
    print(f"  p-value: {entropy_model.pvalues[1]:.6f}")
    print(f"  Effect size (β): {entropy_model.params[1]:.3f}")

    print("\nHypothesis 3: Combined model better than surprisal alone")
    r2_improvement = models['combined']['r2'] - models['surprisal']['r2']
    improvement_significant = r2_improvement > 0.001

    # F-test for nested models
    try:
        f_stat = ((combined_model.ssr - surprisal_model.ssr) / 1) / (combined_model.ssr / combined_model.df_resid)
        f_pvalue = stats.f.sf(f_stat, 1, combined_model.df_resid) if f_stat > 0 else 1.0
    except:
        f_pvalue = 1.0

    print(f"  Result: {'✓ SUPPORTED' if improvement_significant and f_pvalue < 0.05 else '❌ NOT SUPPORTED'}")
    print(f"  R² improvement: {r2_improvement:.4f}")
    print(f"  F-test p-value: {f_pvalue:.6f}")

    print("\nHypothesis 4: Entropy adds unique predictive power beyond surprisal")
    entropy_in_combined_significant = combined_model.pvalues[2] < 0.05
    print(f"  Result: {'✓ SUPPORTED' if entropy_in_combined_significant else '❌ NOT SUPPORTED'}")
    print(f"  Entropy p-value in combined model: {combined_model.pvalues[2]:.6f}")
    print(f"  Entropy coefficient: {combined_model.params[2]:.3f}")

    # Effect size comparison
    print(f"\nEffect Size Comparison (in combined model):")
    surprisal_beta = abs(combined_model.params[1])
    entropy_beta = abs(combined_model.params[2])

    if surprisal_beta > entropy_beta:
        ratio = surprisal_beta / entropy_beta if entropy_beta > 0 else float('inf')
        print(f"  Surprisal effect {ratio:.1f}x stronger than entropy")
    else:
        ratio = entropy_beta / surprisal_beta if surprisal_beta > 0 else float('inf')
        print(f"  Entropy effect {ratio:.1f}x stronger than surprisal")


def create_visualization(models, df):
    """Create visualization comparing model predictions"""
    if not models:
        print("❌ No models for visualization")
        return

    print("\n=== CREATING VISUALIZATIONS ===")

    try:
        # Sample data for visualization (to avoid overplotting)
        sample_size = min(10000, len(df))
        sample_df = df.sample(sample_size, random_state=42)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison: Surprisal vs Entropy vs Combined', fontsize=16)

        # Model performance comparison
        ax1 = axes[0, 0]
        model_names = ['Surprisal', 'Entropy', 'Combined']
        r2_values = [models['surprisal']['r2'], models['entropy']['r2'], models['combined']['r2']]
        bars = ax1.bar(model_names, r2_values, color=['blue', 'green', 'purple'])
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylim(0, max(r2_values) * 1.2 if max(r2_values) > 0 else 0.01)

        # Add value labels on bars
        for bar, value in zip(bars, r2_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(r2_values) * 0.01,
                     f'{value:.4f}', ha='center', va='bottom')

        # Surprisal vs Reading Time
        ax2 = axes[0, 1]
        ax2.scatter(sample_df['pythia_surprisal'], sample_df['reading_time'], alpha=0.3, s=1)
        ax2.set_xlabel('Surprisal (base 10)')
        ax2.set_ylabel('Reading Time (ms)')
        corr = df['pythia_surprisal'].corr(df['reading_time'])
        ax2.set_title(f'Surprisal vs Reading Time (r = {corr:.3f})')

        # Entropy vs Reading Time
        ax3 = axes[1, 0]
        ax3.scatter(sample_df['pythia_entropy'], sample_df['reading_time'], alpha=0.3, s=1, color='green')
        ax3.set_xlabel('Entropy (base 10)')
        ax3.set_ylabel('Reading Time (ms)')
        corr = df['pythia_entropy'].corr(df['reading_time'])
        ax3.set_title(f'Entropy vs Reading Time (r = {corr:.3f})')

        # Combined model residuals
        ax4 = axes[1, 1]
        combined_model = models['combined']['model']
        X_combined = sample_df[['pythia_surprisal', 'pythia_entropy']].values
        y_pred = combined_model.predict(X_combined)
        residuals = sample_df['reading_time'].values - y_pred

        ax4.scatter(y_pred, residuals, alpha=0.3, s=1, color='purple')
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Predicted Reading Time (ms)')
        ax4.set_ylabel('Residuals (ms)')
        ax4.set_title('Combined Model Residuals')

        plt.tight_layout()
        plt.savefig('entropy_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Visualization saved as 'entropy_analysis_results.png'")

    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()


# Legacy compatibility functions
def compute_word_surprisal_batch(sentences, model, tokenizer, device):
    """Legacy function - now processes word by word"""
    return []


def calculate_entropy_fast(sentence, model, tokenizer, device):
    """Legacy function - now processes word by word"""
    return []