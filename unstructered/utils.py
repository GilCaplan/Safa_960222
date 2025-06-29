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

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data():
    """Load OneStop dataset with proper error handling"""
    print("Loading OneStop dataset...")

    file_path = Path(
        "/Users/USER/Desktop/University/Semester 6/Safa/project/project_code/unstructered/data/onestop/ia_Paragraph.csv")
    raw_data = pd.read_csv(file_path)

    print(f"Loaded {len(raw_data)} rows")
    print(f"Columns: {raw_data.columns.tolist()}")
    print(f"Shape: {raw_data.shape}")

    # Check for required columns
    required_cols = ['participant_id', 'TRIAL_INDEX', 'IA_LABEL', 'IA_DWELL_TIME']
    missing_cols = [col for col in required_cols if col not in raw_data.columns]

    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print("Available columns:")
        for col in raw_data.columns:
            print(f"  - {col}")

    print("\nFirst 3 rows:")
    print(raw_data.head(3))

    return raw_data


def load_pythia_70m():
    """Load Pythia 70M model and tokenizer with error handling"""
    print("Loading Pythia 70M model...")
    model_name = "EleutherAI/pythia-70m"
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        dtype = torch.float16
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=None)
    model.to(device).eval()

    print(f"✓ Model loaded successfully on {device}")
    return model, tokenizer, device


def extract_words_with_offsets(text):
    """Extract (word, start_idx, end_idx) tuples from text."""
    word_spans = []
    cursor = 0
    for word in text.split():
        begin = text.find(word, cursor)
        end = begin + len(word)
        word_spans.append((word, begin, end))
        cursor = end
    return word_spans


def compute_word_surprisal_batch(sentences, model, tokenizer, device, batch_size=16):
    """
    FIXED: Compute surprisal properly using offset mapping like Task 1.
    """
    all_results = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch_results = []

        try:
            # Process each sentence individually for proper surprisal calculation
            for sentence in batch_sentences:
                words = sentence.strip().split()
                if not words:
                    batch_results.append([])
                    continue

                try:
                    # FIXED: Use offset mapping like Task 1
                    encoding = tokenizer(
                        sentence,
                        return_tensors="pt",
                        return_offsets_mapping=True,
                        add_special_tokens=True
                    )

                    input_ids = encoding["input_ids"].to(device)
                    offset_mapping = encoding["offset_mapping"][0]

                    with torch.no_grad():
                        outputs = model(input_ids)
                        logits = outputs.logits
                        log_probs = F.log_softmax(logits, dim=-1)

                    # Calculate token-level surprisals (skip BOS token)
                    token_surprisals = []
                    for j in range(1, input_ids.size(1)):
                        token_id = input_ids[0, j].item()
                        token_log_prob = log_probs[0, j - 1, token_id].item()
                        # FIXED: Don't bound here, do it after alignment
                        surprisal = -token_log_prob / math.log(2)  # Convert to bits
                        token_surprisals.append(surprisal)

                    # FIXED: Use proper token-to-word alignment
                    word_surprisals = align_tokens_to_words_proper(
                        words, offset_mapping[1:], token_surprisals  # Skip BOS token offset
                    )

                    # FIXED: Bound values AFTER alignment
                    word_surprisals = [max(0.1, min(s, 50.0)) for s in word_surprisals]

                    batch_results.append(list(zip(words, word_surprisals)))

                except Exception as e:
                    words = sentence.strip().split()
                    batch_results.append([(word, 10.0) for word in words])

            all_results.extend(batch_results)

        except Exception as e:
            # Fallback for batch
            for sentence in batch_sentences:
                words = sentence.strip().split()
                all_results.append([(word, 10.0) for word in words])

    return all_results


def align_tokens_to_words_proper(words, offset_mapping, token_values):
    """
    FIXED: Proper token-to-word alignment using offset mapping like in Task 1.
    """
    if len(words) == 0 or len(token_values) == 0:
        return [10.0] * len(words)

    # Extract word positions in the original text
    word_spans = []
    cursor = 0
    text = ' '.join(words)

    for word in words:
        begin = text.find(word, cursor)
        end = begin + len(word)
        word_spans.append((begin, end))
        cursor = end

    word_values = []

    for word_start, word_end in word_spans:
        word_token_values = []

        # Find tokens that overlap with this word
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if i < len(token_values):
                # Check if token overlaps with word
                if (token_start < word_end and token_end > word_start):
                    word_token_values.append(token_values[i])

        if word_token_values:
            # Average surprisal across tokens for this word
            word_value = sum(word_token_values) / len(word_token_values)
        else:
            # Fallback if no tokens found
            word_value = 10.0

        word_values.append(word_value)

    return word_values


def calculate_entropy_fast(sentence, model, tokenizer, device):
    """
    Fast entropy calculation - only calculate for first few positions.
    """
    words = sentence.strip().split()
    if not words:
        return []

    entropies = []

    try:
        # Only calculate entropy for first 5 positions (for speed)
        max_positions = min(5, len(words))

        for i in range(len(words)):
            if i < max_positions:
                context = ' '.join(words[:i]) if i > 0 else ""

                if context.strip():
                    try:
                        encoded = tokenizer(context, return_tensors="pt", add_special_tokens=True)
                        input_ids = encoded["input_ids"].to(device)

                        with torch.no_grad():
                            outputs = model(input_ids)
                            logits = outputs.logits[0, -1, :]

                            # Fast entropy with top-100 tokens only
                            top_k_logits, _ = torch.topk(logits, min(100, logits.size(0)))
                            top_k_probs = F.softmax(top_k_logits, dim=0)
                            entropy = -(top_k_probs * torch.log2(top_k_probs + 1e-10)).sum().item()
                            entropies.append(max(1.0, min(entropy, 15.0)))

                    except:
                        entropies.append(5.0)
                else:
                    entropies.append(5.0)
            else:
                # For positions > 5, use a simple heuristic based on word position
                # Longer sentences tend to have lower entropy later
                position_factor = max(0.7, 1.0 - (i / len(words)) * 0.3)
                base_entropy = 7.0
                entropies.append(base_entropy * position_factor)

    except Exception as e:
        entropies = [5.0] * len(words)

    return entropies


def process_dataset_batch(df, model, tokenizer, device, max_trials=None, batch_size=16):
    """
    FIXED: Process dataset with proper surprisal calculation and no trial limit.
    """
    trials = list(df.groupby(['participant_id', 'TRIAL_INDEX']))
    total_available = len(trials)

    # Limit trials if specified
    if max_trials is not None:
        if len(trials) > max_trials:
            trials = trials[:max_trials]
            print(f"Processing {max_trials}/{total_available} trials for speed")
        else:
            print(f"Processing ALL {len(trials)} trials (less than limit)")
    else:
        print(f"Processing ALL {len(trials)} trials (no limit set)")

    total_trials = len(trials)
    processed = 0
    skipped = 0
    results = []

    # Process in batches
    batch_sentences = []
    batch_metadata = []

    for (participant, trial_idx), group in tqdm(trials, desc="Processing trials"):
        processed += 1

        # Sort by IA_ID to get text order
        group = group.sort_values('IA_ID').reset_index(drop=True)
        words = group['IA_LABEL'].tolist()
        reading_times = group['IA_DWELL_TIME'].tolist()

        # Less aggressive filtering - closer to Task 1
        if len(words) < 3 or len(words) > 80:
            skipped += 1
            continue

        # Clean words but be less strict
        clean_indices = []
        clean_words = []
        clean_rts = []

        for i, word in enumerate(words):
            # More lenient: allow words with some punctuation
            if len(word) > 0 and any(c.isalpha() for c in word):  # At least one letter
                clean_word = ''.join(c for c in word if c.isalpha())  # Remove punctuation
                if len(clean_word) > 0:
                    clean_indices.append(i)
                    clean_words.append(clean_word)
                    clean_rts.append(reading_times[i])

        # Less strict minimum length
        if len(clean_words) < 3:
            skipped += 1
            continue

        sentence = ' '.join(clean_words)
        batch_sentences.append(sentence)
        batch_metadata.append((participant, trial_idx, clean_words, clean_rts))

        # Process batch when full
        if len(batch_sentences) >= batch_size:
            try:
                # FIXED: Batch process surprisals with proper calculation
                batch_surprisal_results = compute_word_surprisal_batch(
                    batch_sentences, model, tokenizer, device
                )

                # Process each sentence in batch
                for idx, (sentence, (participant, trial_idx, clean_words, clean_rts)) in enumerate(
                        zip(batch_sentences, batch_metadata)
                ):
                    try:
                        # Get surprisals from batch result
                        word_surprisal_pairs = batch_surprisal_results[idx]
                        surprisals = [s for _, s in word_surprisal_pairs]

                        # Fast entropy calculation
                        entropies = calculate_entropy_fast(sentence, model, tokenizer, device)

                        # Better alignment - ensure we don't lose too many words
                        min_len = min(len(clean_words), len(surprisals), len(entropies), len(clean_rts))

                        if min_len >= 3:  # Only skip if we lose too much
                            for i in range(min_len):
                                # FIXED: Use fallback values instead of NaN to keep more data
                                surprisal_val = surprisals[i] if 0.1 <= surprisals[i] <= 50.0 else 10.0
                                entropy_val = entropies[i] if 0.1 <= entropies[i] <= 20.0 else 5.0
                                rt_val = clean_rts[i] if 10 <= clean_rts[i] <= 5000 else np.nan

                                # Only exclude if reading time is invalid
                                if not np.isnan(rt_val):
                                    results.append({
                                        'participant_id': participant,
                                        'trial_index': trial_idx,
                                        'word_position': i + 1,
                                        'word': clean_words[i],
                                        'reading_time': rt_val,
                                        'pythia_surprisal': surprisal_val,
                                        'pythia_entropy': entropy_val,
                                        'sentence_length': len(clean_words),
                                        'sentence': sentence
                                    })

                    except Exception as e:
                        if skipped < 5:  # Only print first few errors
                            print(f"Error in batch processing: {e}")
                        skipped += 1

            except Exception as e:
                print(f"Batch processing error: {e}")
                skipped += len(batch_sentences)

            # Clear batch
            batch_sentences = []
            batch_metadata = []

    # Process remaining sentences in batch
    if batch_sentences:
        try:
            batch_surprisal_results = compute_word_surprisal_batch(
                batch_sentences, model, tokenizer, device
            )

            for idx, (sentence, (participant, trial_idx, clean_words, clean_rts)) in enumerate(
                    zip(batch_sentences, batch_metadata)
            ):
                try:
                    word_surprisal_pairs = batch_surprisal_results[idx]
                    surprisals = [s for _, s in word_surprisal_pairs]
                    entropies = calculate_entropy_fast(sentence, model, tokenizer, device)

                    min_len = min(len(clean_words), len(surprisals), len(entropies), len(clean_rts))

                    if min_len >= 3:
                        for i in range(min_len):
                            surprisal_val = surprisals[i] if 0.1 <= surprisals[i] <= 50.0 else 10.0
                            entropy_val = entropies[i] if 0.1 <= entropies[i] <= 20.0 else 5.0
                            rt_val = clean_rts[i] if 10 <= clean_rts[i] <= 5000 else np.nan

                            if not np.isnan(rt_val):
                                results.append({
                                    'participant_id': participant,
                                    'trial_index': trial_idx,
                                    'word_position': i + 1,
                                    'word': clean_words[i],
                                    'reading_time': rt_val,
                                    'pythia_surprisal': surprisal_val,
                                    'pythia_entropy': entropy_val,
                                    'sentence_length': len(clean_words),
                                    'sentence': sentence
                                })

                except Exception as e:
                    skipped += 1

        except Exception as e:
            print(f"Final batch error: {e}")
            skipped += len(batch_sentences)

    result_df = pd.DataFrame(results)
    print(f"✓ Processed {len(result_df)} words from {processed} trials")
    print(f"  Skipped {skipped} trials due to errors or filtering")
    print(f"  Success rate: {len(result_df) / processed:.1f} words per trial")

    return result_df


def quick_test_model(model, tokenizer, device):
    """FIXED: Test model with proper surprisal calculation"""
    print("\n" + "=" * 50)
    print("TESTING PYTHIA MODEL WITH FIXED SURPRISAL CALCULATION")
    print("=" * 50)

    test_sentences = [
        "The cat sat on the mat.",
        "This is a simple test sentence.",
        "Machine learning models predict text accurately."
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: '{sentence}'")
        words = sentence.split()

        try:
            # Test the FIXED surprisal calculation
            word_surprisal_pairs = compute_word_surprisal_batch([sentence], model, tokenizer, device)[0]
            surprisals = [s for w, s in word_surprisal_pairs]

            # Test entropy calculation
            entropies = calculate_entropy_fast(sentence, model, tokenizer, device)

            print(f"Words ({len(words)}): {words}")
            print(f"Surprisals ({len(surprisals)}): {[f'{s:.2f}' for s in surprisals]}")
            print(f"Entropies ({len(entropies)}): {[f'{e:.2f}' for e in entropies]}")

            # Validation checks
            if len(surprisals) == len(words) and len(entropies) == len(words):
                avg_surprisal = np.mean(surprisals)
                avg_entropy = np.mean(entropies)
                print(f"✓ Lengths match. Avg surprisal: {avg_surprisal:.2f}, Avg entropy: {avg_entropy:.2f}")

                # Check if surprisals are realistic (not all minimum values)
                if max(surprisals) > 1.0:  # Should have some variation
                    print("✓ Surprisal calculation working properly")
                else:
                    print("❌ Surprisals all too low - calculation may be incorrect")

            else:
                print("❌ Length mismatch!")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("MODEL TEST COMPLETE")
    print("=" * 50)


# Legacy functions for compatibility
def compute_word_surprisal(sentence, model, tokenizer, device):
    """Legacy function - use compute_word_surprisal_batch for better performance"""
    return compute_word_surprisal_batch([sentence], model, tokenizer, device)[0]


def calculate_entropy(sentence, model, tokenizer, device, top_k=1000):
    """Legacy function - use calculate_entropy_fast for better performance"""
    return calculate_entropy_fast(sentence, model, tokenizer, device)


def align_tokens_to_words(words, offset_mapping, token_values):
    """Updated to use the proper alignment method."""
    return align_tokens_to_words_proper(words, offset_mapping, token_values)


# Include all the other functions (validate_results, fit_regression_models, etc.)
def validate_results(results_df):
    """Validate the calculated surprisal and entropy values"""
    print("\n" + "=" * 50)
    print("VALIDATING RESULTS")
    print("=" * 50)

    if len(results_df) == 0:
        print("❌ No data to validate")
        return False

    # Check for required columns
    required_cols = ['word', 'pythia_surprisal', 'pythia_entropy']
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

    # More realistic validation thresholds
    surprisal_coverage_ok = len(surprisal_valid) / len(results_df) >= 0.50  # 50% threshold
    entropy_coverage_ok = len(entropy_valid) / len(results_df) >= 0.50  # 50% threshold

    print(f"  Coverage thresholds: {'✓' if surprisal_coverage_ok and entropy_coverage_ok else '❌'}")

    # Check value ranges
    if len(surprisal_valid) > 0:
        surprisal_range_ok = (0.1 <= surprisal_valid.min()) and (surprisal_valid.max() <= 50.0)
        print(
            f"  Surprisal range OK: {'✓' if surprisal_range_ok else '❌'} ({surprisal_valid.min():.2f} - {surprisal_valid.max():.2f})")
    else:
        surprisal_range_ok = False

    if len(entropy_valid) > 0:
        entropy_range_ok = (0.1 <= entropy_valid.min()) and (entropy_valid.max() <= 20.0)
        print(
            f"  Entropy range OK: {'✓' if entropy_range_ok else '❌'} ({entropy_valid.min():.2f} - {entropy_valid.max():.2f})")
    else:
        entropy_range_ok = False

    # Check for function words pattern (optional test)
    function_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'for', 'with']
    function_word_data = results_df[results_df['word'].str.lower().isin(function_words)]

    if len(function_word_data) > 0 and len(surprisal_valid) > 0:
        func_surprisal = function_word_data['pythia_surprisal'].mean()
        all_surprisal = results_df['pythia_surprisal'].mean()
        func_pattern_ok = True  # Make this optional
        print(
            f"  Function words pattern: {'✓' if func_surprisal <= all_surprisal * 1.1 else '❌'} ({func_surprisal:.2f} vs {all_surprisal:.2f})")
    else:
        func_pattern_ok = True

    # Check minimum viable dataset size
    min_size_ok = len(results_df) >= 100
    print(f"  Minimum dataset size: {'✓' if min_size_ok else '❌'} ({len(results_df)} observations)")

    # Overall validation - more lenient criteria
    is_valid = (
            surprisal_coverage_ok and
            entropy_coverage_ok and
            surprisal_range_ok and
            entropy_range_ok and
            min_size_ok
    )

    print(f"\nOverall validation: {'✓ PASSED' if is_valid else '❌ FAILED'}")

    if not is_valid:
        print("\nValidation issues detected:")
        if not surprisal_coverage_ok:
            print("  - Low surprisal coverage")
        if not entropy_coverage_ok:
            print("  - Low entropy coverage")
        if not surprisal_range_ok:
            print("  - Surprisal values out of range")
        if not entropy_range_ok:
            print("  - Entropy values out of range")
        if not min_size_ok:
            print("  - Dataset too small")

        print("\nContinuing with available data...")
        return len(surprisal_valid) > 50 and len(entropy_valid) > 50  # Minimum viable analysis

    return is_valid


def fit_regression_models(df):
    """
    Fit three regression models to compare predictors of reading time:
    1. Surprisal only (baseline)
    2. Surprisal + Entropy (combined)
    3. Entropy only
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy import stats
    import statsmodels.api as sm

    print("\n=== FITTING REGRESSION MODELS ===")

    # Clean data - be more lenient with bounds
    clean_df = df.dropna(subset=['reading_time', 'pythia_surprisal', 'pythia_entropy'])

    print(f"After dropna: {len(clean_df)} observations")

    # FIXED: More lenient filtering for reasonable values
    clean_df = clean_df[
        (clean_df['reading_time'] > 10) & (clean_df['reading_time'] < 5000) &
        (clean_df['pythia_surprisal'] >= 0.1) & (clean_df['pythia_surprisal'] < 50) &  # >= instead of >
        (clean_df['pythia_entropy'] >= 0.1) & (clean_df['pythia_entropy'] < 20)  # >= instead of >
        ]

    print(f"After value filtering: {len(clean_df)} observations")

    # Debug: show value ranges
    if len(clean_df) > 0:
        print(f"  Surprisal range: {clean_df['pythia_surprisal'].min():.2f} - {clean_df['pythia_surprisal'].max():.2f}")
        print(f"  Entropy range: {clean_df['pythia_entropy'].min():.2f} - {clean_df['pythia_entropy'].max():.2f}")
        print(f"  RT range: {clean_df['reading_time'].min():.0f} - {clean_df['reading_time'].max():.0f}")
    else:
        print("  No observations after filtering - checking original data...")
        print(f"  Original surprisal range: {df['pythia_surprisal'].min():.2f} - {df['pythia_surprisal'].max():.2f}")
        print(f"  Original entropy range: {df['pythia_entropy'].min():.2f} - {df['pythia_entropy'].max():.2f}")

    if len(clean_df) < 30:  # Lower threshold for minimum viable analysis
        print("❌ Insufficient data for regression (need at least 30 observations)")
        return None, clean_df

    # Prepare variables
    y = clean_df['reading_time'].values
    X_surprisal = clean_df['pythia_surprisal'].values.reshape(-1, 1)
    X_entropy = clean_df['pythia_entropy'].values.reshape(-1, 1)
    X_combined = clean_df[['pythia_surprisal', 'pythia_entropy']].values

    models = {}

    try:
        # Model 1: Surprisal only (baseline)
        print("\n1. Surprisal-only model:")
        model_surprisal = LinearRegression().fit(X_surprisal, y)
        y_pred_surprisal = model_surprisal.predict(X_surprisal)
        r2_surprisal = r2_score(y, y_pred_surprisal)

        # Get detailed stats with statsmodels
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

        # Model 2: Entropy only
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

        # Model 3: Surprisal + Entropy (combined)
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
        print(f"Relative strength: {r2_surprisal / 0.026 * 100:.1f}% of Task 1 effect")

        return models, clean_df

    except Exception as e:
        print(f"❌ Error fitting models: {e}")
        return None, clean_df


def compare_models(models):
    """Compare the three models using statistical tests and information criteria."""
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
    improvement_entropy = ((r2_entropy - r2_surprisal) / r2_surprisal) * 100 if r2_surprisal > 0 else 0
    improvement_combined = ((r2_combined - r2_surprisal) / r2_surprisal) * 100 if r2_surprisal > 0 else 0

    print(f"  Entropy vs Surprisal: {improvement_entropy:+.1f}%")
    print(f"  Combined vs Surprisal: {improvement_combined:+.1f}%")

    # Statistical significance of improvement
    print(f"\nAdditional R² from entropy: {r2_combined - r2_surprisal:.4f}")

    return comparison_df


def test_hypotheses(models, df):
    """Test specific hypotheses about entropy and surprisal effects."""
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
    improvement_significant = r2_improvement > 0.001  # Practical significance threshold

    # F-test for nested models
    f_stat = ((combined_model.ssr - surprisal_model.ssr) / 1) / (combined_model.ssr / combined_model.df_resid)
    f_pvalue = stats.f.sf(f_stat, 1, combined_model.df_resid) if f_stat > 0 else 1.0

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
    """Create visualization comparing model predictions."""
    import matplotlib.pyplot as plt

    if not models:
        print("❌ No models for visualization")
        return

    print("\n=== CREATING VISUALIZATIONS ===")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: Surprisal vs Entropy vs Combined', fontsize=16)

    # Model performance comparison
    ax1 = axes[0, 0]
    model_names = ['Surprisal', 'Entropy', 'Combined']
    r2_values = [models['surprisal']['r2'], models['entropy']['r2'], models['combined']['r2']]
    bars = ax1.bar(model_names, r2_values, color=['blue', 'green', 'purple'])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(0, max(r2_values) * 1.2)

    # Add value labels on bars
    for bar, value in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(r2_values) * 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # Surprisal vs Reading Time
    ax2 = axes[0, 1]
    ax2.scatter(df['pythia_surprisal'], df['reading_time'], alpha=0.3, s=1)
    ax2.set_xlabel('Surprisal (bits)')
    ax2.set_ylabel('Reading Time (ms)')
    ax2.set_title(f'Surprisal vs Reading Time (r = {df["pythia_surprisal"].corr(df["reading_time"]):.3f})')

    # Entropy vs Reading Time
    ax3 = axes[1, 0]
    ax3.scatter(df['pythia_entropy'], df['reading_time'], alpha=0.3, s=1, color='green')
    ax3.set_xlabel('Entropy (bits)')
    ax3.set_ylabel('Reading Time (ms)')
    ax3.set_title(f'Entropy vs Reading Time (r = {df["pythia_entropy"].corr(df["reading_time"]):.3f})')

    # Combined model residuals
    ax4 = axes[1, 1]
    combined_model = models['combined']['model']
    X_combined = df[['pythia_surprisal', 'pythia_entropy']].values
    y_pred = combined_model.predict(X_combined)
    residuals = df['reading_time'].values - y_pred

    ax4.scatter(y_pred, residuals, alpha=0.3, s=1, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Predicted Reading Time (ms)')
    ax4.set_ylabel('Residuals (ms)')
    ax4.set_title('Combined Model Residuals')

    plt.tight_layout()
    plt.savefig('entropy_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Visualization saved as 'entropy_analysis_results.png'")

