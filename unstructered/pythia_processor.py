import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import math
import subprocess
import urllib.request
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import traceback
import pickle
import kenlm
from datasets import load_dataset

print("✓ Hugging Face datasets available")
HF_DATASETS_AVAILABLE = True

plt.style.use('default')
sns.set_palette("husl")

class PythiaModel:
    """Pythia-70M model using offset-based surprisal calculation."""

    def __init__(self, size_name="70m"):
        model_name = f"EleutherAI/pythia-{size_name}"
        print(f"Loading Pythia-{size_name} model...")

        # Set device
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        print(f"Using device: {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ Pythia model loaded successfully!")

    def _extract_words_with_offsets(self, text):
        """Extract (word, start_idx, end_idx) tuples from text."""
        word_spans = []
        cursor = 0
        for word in text.split():
            begin = text.find(word, cursor)
            end = begin + len(word)
            word_spans.append((word, begin, end))
            cursor = end
        return word_spans

    def get_surprisal_and_probability(self, sentence):
        """
        Compute word-level surprisal (base-10 log) and probabilities.
        FIXED: NO CLAMPING - keep natural values
        """
        try:
            words_with_offsets = self._extract_words_with_offsets(sentence)
            encoded = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True,
                                     add_special_tokens=True)
            input_ids = encoded["input_ids"].to(self.device)
            offset_map = encoded["offset_mapping"][0]  # shape: [seq_len, 2]

            with torch.no_grad():
                logits = self.model(input_ids).logits
                log_probs = torch.log_softmax(logits, dim=-1)

            surprisals = []
            for i in range(1, input_ids.size(1)):
                token = input_ids[0, i].item()
                token_log_prob = log_probs[0, i - 1, token]
                surprisal_val = -token_log_prob.item() / math.log(10)  # log base 10
                surprisals.append(surprisal_val)

            # Align tokens to words using offset mapping
            word_surprisals = []
            token_pointer = 0
            total_tokens = len(surprisals)

            for word, w_start, w_end in words_with_offsets:
                acc_surprisal = 0.0
                token_count = 0

                while token_pointer < total_tokens:
                    t_start, t_end = offset_map[token_pointer + 1].tolist()
                    if t_start >= w_end:
                        break
                    if t_end <= w_start:
                        token_pointer += 1
                        continue
                    acc_surprisal += surprisals[token_pointer]  # SUM - no clamping
                    token_count += 1
                    token_pointer += 1

                # FIXED: NO CLAMPING - keep natural values
                if token_count > 0:
                    word_surprisal = acc_surprisal
                    # Check for NaN/inf and use reasonable fallback
                    if math.isnan(word_surprisal) or math.isinf(word_surprisal):
                        word_surprisal = 8.0
                else:
                    word_surprisal = 8.0  # Reasonable fallback for unmatched words

                word_surprisals.append(word_surprisal)

            # Calculate probabilities: 10^(-surprisal)
            word_probabilities = []
            for s in word_surprisals:
                prob = 10 ** (-s)
                # Ensure probability is valid
                if math.isnan(prob) or math.isinf(prob) or prob <= 0:
                    prob = 1e-8
                word_probabilities.append(prob)

            return word_surprisals, word_probabilities

        except Exception as e:
            print(f"Error computing Pythia surprisal: {e}")
            fallback_len = len(sentence.split())
            return [8.0] * fallback_len, [1e-8] * fallback_len


def load_and_preprocess_data(file_path):
    """Load and preprocess OneStop dataset"""
    print("Loading OneStop dataset...")
    df = pd.read_csv(file_path)

    # Use correct columns
    word_col = 'IA_LABEL'
    rt_col = 'IA_DWELL_TIME'
    participant_col = 'participant_id'
    trial_col = 'TRIAL_INDEX'

    # Basic preprocessing
    df = df.dropna(subset=[word_col, rt_col])
    df = df[df[rt_col] > 0]
    df = df[df[word_col].str.len() > 0]
    df = df[df[word_col].str.isalpha()]

    # Rename columns
    df = df.rename(columns={
        word_col: 'WORD',
        rt_col: 'IA_DWELL_TIME',
        participant_col: 'PARTICIPANT',
        trial_col: 'TRIAL'
    })

    # Sort by reading order
    sort_columns = ['PARTICIPANT', 'TRIAL']
    if 'IA_ID' in df.columns:
        sort_columns.append('IA_ID')
    elif 'INTEREST_AREA_FIXATION_SEQUENCE' in df.columns:
        sort_columns.append('INTEREST_AREA_FIXATION_SEQUENCE')
    elif 'IA_FIRST_FIXATION_TIME' in df.columns:
        sort_columns.append('IA_FIRST_FIXATION_TIME')

    df = df.sort_values(sort_columns)

    # Remove outliers
    q99 = df['IA_DWELL_TIME'].quantile(0.99)
    q01 = df['IA_DWELL_TIME'].quantile(0.01)
    df = df[(df['IA_DWELL_TIME'] >= q01) & (df['IA_DWELL_TIME'] <= q99)]

    print(f"Dataset loaded: {len(df)} words from {df['PARTICIPANT'].nunique()} participants")
    return df


def compute_surprisals(test_df, kenlm_model, pythia_model):
    """
    Compute surprisals using both models
    FIXED: NO CLAMPING in filtering - use natural ranges
    """
    print("Computing surprisal values...")

    surprisal_data = []
    total_groups = len(test_df.groupby(['PARTICIPANT', 'TRIAL']))
    processed = 0

    for (participant, trial), group in test_df.groupby(['PARTICIPANT', 'TRIAL']):
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_groups} trials")

        group = group.reset_index(drop=True)
        words = group['WORD'].tolist()
        reading_times = group['IA_DWELL_TIME'].tolist()

        if len(words) < 3 or len(words) > 100:
            continue

        sentence = ' '.join(words)

        try:
            # Get surprisals from both models
            kenlm_surprisals, kenlm_probs = kenlm_model.get_surprisal_and_probability(sentence)
            pythia_surprisals, pythia_probs = pythia_model.get_surprisal_and_probability(sentence)

            # Ensure alignment
            min_len = min(len(words), len(kenlm_surprisals), len(pythia_surprisals), len(reading_times))

            if min_len < 3:
                continue

            # Store data with more lenient filtering - NO CLAMPING
            for i in range(min_len):
                # Check for valid values (not NaN/inf) but don't clamp
                kenlm_surp = kenlm_surprisals[i]
                pythia_surp = pythia_surprisals[i]
                rt = reading_times[i]

                # Only exclude if values are clearly invalid
                if (not math.isnan(kenlm_surp) and not math.isinf(kenlm_surp) and
                        not math.isnan(pythia_surp) and not math.isinf(pythia_surp) and
                        not math.isnan(rt) and not math.isinf(rt) and
                        kenlm_surp > 0 and pythia_surp > 0 and rt > 0):
                    surprisal_data.append({
                        'PARTICIPANT': participant,
                        'TRIAL': trial,
                        'WORD': words[i],
                        'WORD_INDEX': i,
                        'IA_DWELL_TIME': rt,
                        'TRIGRAM_SURPRISAL': kenlm_surp,  # Natural values
                        'PYTHIA_SURPRISAL': pythia_surp,  # Natural values
                        'TRIGRAM_PROBABILITY': kenlm_probs[i],
                        'PYTHIA_PROBABILITY': pythia_probs[i]
                    })

        except Exception as e:
            print(f"Error processing trial {trial}: {e}")
            continue

    result_df = pd.DataFrame(surprisal_data)
    print(f"✓ Computed surprisals for {len(result_df)} words")
    print(
        f"Trigram surprisal range: {result_df['TRIGRAM_SURPRISAL'].min():.2f} - {result_df['TRIGRAM_SURPRISAL'].max():.2f}")
    print(
        f"Pythia surprisal range: {result_df['PYTHIA_SURPRISAL'].min():.2f} - {result_df['PYTHIA_SURPRISAL'].max():.2f}")
    return result_df



def calculate_entropy(sentence, model, tokenizer, device):
    """Simple entropy calculation that matches word alignment with surprisal"""
    import torch
    import torch.nn.functional as F

    # Use same approach as pythia_processor.py for word extraction and alignment
    words_with_offsets = []
    cursor = 0
    for word in sentence.split():
        begin = sentence.find(word, cursor)
        end = begin + len(word)
        words_with_offsets.append((word, begin, end))
        cursor = end

    # Tokenize with offset mapping (same as pythia_processor.py)
    encoded = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    offset_map = encoded["offset_mapping"][0]

    # Get model predictions
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # Remove batch dimension

    # Calculate entropy for each token
    token_entropies = []
    for i in range(len(logits)):
        probs = F.softmax(logits[i], dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        token_entropies.append(entropy.item())

    # Align tokens to words using same method as pythia_processor.py
    word_entropies = []
    token_pointer = 0
    total_tokens = len(token_entropies)

    for word, w_start, w_end in words_with_offsets:
        acc_entropy = 0.0
        token_count = 0

        while token_pointer < total_tokens:
            if token_pointer + 1 >= len(offset_map):
                break
            t_start, t_end = offset_map[token_pointer + 1].tolist()
            if t_start >= w_end:
                break
            if t_end <= w_start:
                token_pointer += 1
                continue
            acc_entropy += token_entropies[token_pointer]
            token_count += 1
            token_pointer += 1

        if token_count > 0:
            word_entropy = acc_entropy
        else:
            word_entropy = 5.0  # Default fallback

        word_entropies.append(word_entropy)

    return word_entropies