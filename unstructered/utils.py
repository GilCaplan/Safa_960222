import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from tqdm import tqdm
import numpy as np
import math
warnings.filterwarnings('ignore')


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

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=None )
    model.to(device).eval()

    print(f"✓ Model loaded successfully on {device}")
    return model, tokenizer, device


def extract_words_with_offsets(text):
    """Extract words with their character positions in the text"""
    word_spans = []
    cursor = 0
    for word in text.split():
        begin = text.find(word, cursor)
        finish = begin + len(word)
        word_spans.append((word, begin, finish))
        cursor = finish
    return word_spans


def compute_word_surprisal(text, model, tokenizer, device):
    """Compute word-level surprisal returning list of (word, surprisal) tuples"""
    if not text or not text.strip():
        return []

    try:
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encoded["input_ids"].to(device)
        offset_map = encoded["offset_mapping"][0]

        with torch.no_grad():
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        # Calculate token surprisals
        surprisals = []
        for i in range(1, input_ids.size(1)):
            token = input_ids[0, i].item()
            token_log_prob = log_probs[0, i - 1, token]
            surprisal_val = -token_log_prob.item() / math.log(2)  # Convert to bits
            surprisal_val = max(0.1, min(surprisal_val, 50.0))  # Clamp values
            surprisals.append(surprisal_val)

        # Get word offsets and align tokens to words
        word_offsets = extract_words_with_offsets(text)
        word_level_surprisal = []
        token_pointer = 0
        total_tokens = len(surprisals)

        for word, w_start, w_end in word_offsets:
            acc_surprisal = 0.0
            while token_pointer < total_tokens:
                if token_pointer + 1 >= len(offset_map):
                    break
                t_start, t_end = offset_map[token_pointer + 1].tolist()
                if t_start >= w_end:
                    break
                if t_end <= w_start:
                    token_pointer += 1
                    continue
                acc_surprisal += surprisals[token_pointer]
                token_pointer += 1

            word_level_surprisal.append((word, acc_surprisal if acc_surprisal > 0 else 5.0))

        return word_level_surprisal

    except Exception as e:
        print(f"Error computing surprisal for '{text[:30]}...': {e}")
        # Return empty list on error
        return []


def calculate_surprisal(text, model, tokenizer, device):
    """Calculate word-level surprisal returning list of surprisal values"""
    if not text or not text.strip():
        return []

    text = text.strip()
    words = text.split()
    if len(words) == 0:
        return []

    try:
        word_surprisal_pairs = compute_word_surprisal(text, model, tokenizer, device)
        return [surprisal for word, surprisal in word_surprisal_pairs]
    except Exception as e:
        print(f"Error computing surprisal for '{text[:30]}...': {e}")
        return [5.0] * len(words)


def calculate_entropy(text, model, tokenizer, device):
    """Calculate word-level entropy using offset mapping for proper alignment"""
    if not text or not text.strip():
        return []

    text = text.strip()
    words = text.split()
    if len(words) == 0:
        return []

    try:
        # Tokenize with offset mapping
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encoded["input_ids"].to(device)
        offset_map = encoded["offset_mapping"][0]

        with torch.no_grad():
            logits = model(input_ids).logits

        # Calculate token-level entropy (skip first special token)
        entropies = []
        for i in range(1, input_ids.size(1)):
            # Get probability distribution at previous position (predicting current token)
            prev_logits = logits[0, i - 1]
            probs = torch.softmax(prev_logits, dim=-1)

            # Convert to numpy for entropy calculation
            prob_array = probs.cpu().numpy()

            # Filter very small probabilities to avoid numerical issues
            significant_probs = prob_array[prob_array > 1e-8]

            if len(significant_probs) > 0:
                # Normalize probabilities
                significant_probs = significant_probs / np.sum(significant_probs)
                # Calculate entropy in bits
                entropy = -np.sum(significant_probs * np.log2(significant_probs + 1e-10))
                entropy = max(0.1, min(entropy, 20.0))  # Clamp
            else:
                entropy = 0.1

            entropies.append(entropy)

        # Get words with their character offsets
        word_offsets = extract_words_with_offsets(text)

        # Align tokens to words using offset mapping
        word_level_entropy = []
        token_pointer = 0
        total_tokens = len(entropies)

        for word, w_start, w_end in word_offsets:
            acc_entropy = 0.0
            tokens_used = 0

            while token_pointer < total_tokens:
                # +1 because first token is special token (BOS/CLS)
                if token_pointer + 1 >= len(offset_map):
                    break

                t_start, t_end = offset_map[token_pointer + 1].tolist()

                # Token starts after word ends
                if t_start >= w_end:
                    break

                # Token ends before word starts
                if t_end <= w_start:
                    token_pointer += 1
                    continue

                # Token overlaps with word
                acc_entropy += entropies[token_pointer]
                tokens_used += 1
                token_pointer += 1

            # For entropy, take average of token entropies (unlike surprisal which sums)
            word_entropy = acc_entropy / tokens_used if tokens_used > 0 else 10.0
            word_level_entropy.append(word_entropy)

        # Ensure we have the right number of entropies
        while len(word_level_entropy) < len(words):
            word_level_entropy.append(10.0)

        return word_level_entropy[:len(words)]

    except Exception as e:
        print(f"Error calculating entropy for '{text[:30]}...': {e}")
        return [10.0] * len(words)


def process_dataset_surprisal_entropy(df, model, tokenizer, device, sample_size=None):
    """
    Process OneStop dataset using explode pattern for efficiency
    """
    print("Processing dataset for surprisal and entropy...")

    if df is None or len(df) == 0:
        print("Empty or invalid DataFrame")
        return pd.DataFrame()

    required_cols = ['participant_id', 'TRIAL_INDEX', 'IA_LABEL']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    print(f"Original dataset: {len(df)} rows")

    print("Creating sentences from trials...")
    sentence_df = []

    grouped = df.groupby(['participant_id', 'TRIAL_INDEX'])
    for (participant_id, trial_index), group in tqdm(grouped, desc="Creating sentences"):
        if 'IA_INDEX' in group.columns:
            group = group.sort_values('IA_INDEX')

        # Extract words and reading times
        words = group['IA_LABEL'].astype(str).tolist()
        words = [w.strip() for w in words if w and w != 'nan' and len(w.strip()) > 0]

        if len(words) < 2 or len(words) > 50:  # Skip very short/long sentences
            continue

        sentence = ' '.join(words)

        # Store reading times aligned with words
        reading_times = group['IA_DWELL_TIME'].tolist()[:len(words)]

        sentence_df.append({
            'participant_id': participant_id,
            'TRIAL_INDEX': trial_index,
            'sentence': sentence,
            'words': words,
            'reading_times': reading_times
        })

    sentence_df = pd.DataFrame(sentence_df)
    print(f"Created {len(sentence_df)} sentences from trials")

    if sample_size:
        print(f"Taking sample of {sample_size} sentences...")
        sentence_df = sentence_df.head(sample_size)

    print("Computing surprisal values...")
    sentence_df['surprisal_pairs'] = sentence_df['sentence'].apply(
        lambda x: compute_word_surprisal(x, model, tokenizer, device)
    )

    print("Computing entropy values...")
    sentence_df['entropy_values'] = sentence_df['sentence'].apply(
        lambda x: calculate_entropy(x, model, tokenizer, device)
    )

    print("Flattening results...")
    sentence_df = sentence_df.explode('surprisal_pairs', ignore_index=True)

    valid_rows = sentence_df['surprisal_pairs'].notna()
    sentence_df = sentence_df[valid_rows].copy()

    if len(sentence_df) > 0:
        sentence_df[['word', 'pythia_surprisal']] = pd.DataFrame(
            sentence_df['surprisal_pairs'].tolist(),
            index=sentence_df.index
        )

        sentence_df['word_position'] = sentence_df.groupby(['participant_id', 'TRIAL_INDEX']).cumcount()

        entropy_results = []
        for _, row in sentence_df.iterrows():
            word_pos = row['word_position']
            entropy_vals = row['entropy_values']
            if isinstance(entropy_vals, list) and word_pos < len(entropy_vals):
                entropy_results.append(entropy_vals[word_pos])
            else:
                entropy_results.append(10.0)  # Default entropy

        sentence_df['pythia_entropy'] = entropy_results

        rt_results = []
        for _, row in sentence_df.iterrows():
            word_pos = row['word_position']
            rt_vals = row['reading_times']
            if isinstance(rt_vals, list) and word_pos < len(rt_vals):
                rt_results.append(rt_vals[word_pos])
            else:
                rt_results.append(np.nan)

        sentence_df['IA_DWELL_TIME'] = rt_results

        result_cols = ['participant_id', 'TRIAL_INDEX', 'word_position', 'word',
                       'IA_DWELL_TIME', 'pythia_surprisal', 'pythia_entropy']
        results_df = sentence_df[result_cols].copy()

        print(f"\n✓ Processing complete!")
        print(f"Results: {len(results_df)} word observations")

        valid_surprisal = results_df['pythia_surprisal'].dropna()
        valid_entropy = results_df['pythia_entropy'].dropna()

        if len(valid_surprisal) > 0:
            print(f"\nSurprisal statistics:")
            print(f"  Count: {len(valid_surprisal)}")
            print(f"  Mean: {valid_surprisal.mean():.2f}")
            print(f"  Range: {valid_surprisal.min():.2f} - {valid_surprisal.max():.2f}")

        if len(valid_entropy) > 0:
            print(f"\nEntropy statistics:")
            print(f"  Count: {len(valid_entropy)}")
            print(f"  Mean: {valid_entropy.mean():.2f}")
            print(f"  Range: {valid_entropy.min():.2f} - {valid_entropy.max():.2f}")

        print(f"\nSample results:")
        print(results_df[['word', 'pythia_surprisal', 'pythia_entropy']].head(50))

        return results_df
    else:
        print("No valid results generated!")
        return pd.DataFrame()


def quick_test_model(model, tokenizer, device):
    """Quick test to verify model functionality with offset mapping"""
    print("\n" + "=" * 50)
    print("TESTING PYTHIA MODEL WITH OFFSET MAPPING")
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
            # Test offset mapping alignment
            encoding = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            offsets = encoding["offset_mapping"][0]

            print(f"Words ({len(words)}): {words}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Offsets: {offsets.tolist()}")

            # Show word-token alignment
            word_offsets = extract_words_with_offsets(sentence)
            print(f"Word offsets: {word_offsets}")

            # Calculate surprisals using both methods
            surprisal_pairs = compute_word_surprisal(sentence, model, tokenizer, device)
            surprisals = [s for w, s in surprisal_pairs]  # Extract just the surprisal values
            entropies = calculate_entropy(sentence, model, tokenizer, device)

            print(f"Surprisals ({len(surprisals)}): {[f'{s:.2f}' for s in surprisals]}")
            print(f"Entropies ({len(entropies)}): {[f'{e:.2f}' for e in entropies]}")

            # Validation checks
            if len(surprisals) == len(words) and len(entropies) == len(words):
                avg_surprisal = np.mean(surprisals)
                avg_entropy = np.mean(entropies)
                print(f"✓ Lengths match. Avg surprisal: {avg_surprisal:.2f}, Avg entropy: {avg_entropy:.2f}")

                # Check offset alignment worked
                print("✓ Offset mapping alignment successful")
            else:
                print("Length mismatch!")

        except Exception as e:
            print(f"❌ Error: {e}")

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

    # Check value ranges
    if len(surprisal_valid) > 0:
        surprisal_range_ok = (0.1 <= surprisal_valid.min()) and (surprisal_valid.max() <= 50.0)
        print(
            f"  Surprisal range OK: {'✓' if surprisal_range_ok else '❌'} ({surprisal_valid.min():.2f} - {surprisal_valid.max():.2f})")

    if len(entropy_valid) > 0:
        entropy_range_ok = (0.1 <= entropy_valid.min()) and (entropy_valid.max() <= 20.0)
        print(
            f"  Entropy range OK: {'✓' if entropy_range_ok else '❌'} ({entropy_valid.min():.2f} - {entropy_valid.max():.2f})")

    # Check for function words pattern
    function_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'for', 'with']
    function_word_data = results_df[results_df['word'].str.lower().isin(function_words)]

    if len(function_word_data) > 0:
        func_surprisal = function_word_data['pythia_surprisal'].mean()
        all_surprisal = results_df['pythia_surprisal'].mean()
        print(
            f"  Function words lower surprisal: {'✓' if func_surprisal < all_surprisal else '❌'} ({func_surprisal:.2f} vs {all_surprisal:.2f})")

    # Overall validation
    is_valid = (len(surprisal_valid) > len(results_df) * 0.8 and
                len(entropy_valid) > len(results_df) * 0.8)

    print(f"\nOverall validation: {'✓ PASSED' if is_valid else '❌ FAILED'}")
    return is_valid