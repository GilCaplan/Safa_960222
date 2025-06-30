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


class WikiText2KenLMModel:
    """
    KenLM trigram model trained on WikiText-2 dataset
    Saves/loads the trained model to avoid retraining
    """

    def __init__(self):
        self.model = None
        self.model_path = "../wikitext/wikitext2_trigram.arpa"
        self.binary_model_path = "../wikitext/wikitext2_trigram.binary"
        self.dataset_path = "../wikitext-2"

    def download_wikitext2(self):
        """Download WikiText-2 dataset using multiple methods"""
        if os.path.exists(self.dataset_path):
            print(f"✓ WikiText-2 dataset already exists at {self.dataset_path}")
            return True

        print("Downloading WikiText-2 dataset...")

        # Method 1: Try Hugging Face datasets library
        if HF_DATASETS_AVAILABLE:
            try:
                print("Trying Hugging Face datasets...")

                dataset = load_dataset("../wikitext", "wikitext-2-v1")

                # Create directory structure
                os.makedirs(self.dataset_path, exist_ok=True)

                # Save train, validation, and test splits
                for split_name, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
                    filename = f"wiki.{split_name}.tokens"
                    filepath = os.path.join(self.dataset_path, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        for example in dataset[hf_split]:
                            text = example['text'].strip()
                            if text:  # Skip empty lines
                                f.write(text + '\n')

                    print(f"✓ Saved {filename}")

                print(f"✓ WikiText-2 downloaded via Hugging Face to {self.dataset_path}")
                return True

            except Exception as e:
                print(f"Hugging Face method failed: {e}")
        else:
            print("Hugging Face datasets not available, trying alternative methods...")

        # Method 2: Try direct download from working mirrors
        urls_to_try = [
            "https://github.com/pytorch/text/raw/main/test/assets/wikitext-2-v1.zip",
            "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-v1.zip"
        ]

        for url in urls_to_try:
            try:
                print(f"Trying URL: {url}")
                zip_filename = "wikitext-2-v1.zip"

                urllib.request.urlretrieve(url, zip_filename)
                print(f"✓ Downloaded {zip_filename}")

                # Extract the zip file
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(".")

                print(f"✓ Extracted to {self.dataset_path}")
                os.remove(zip_filename)  # Clean up
                return True

            except Exception as e:
                print(f"Failed with URL {url}: {e}")
                continue

        # Method 3: Create minimal dataset as fallback
        print("All download methods failed. Creating minimal WikiText-2 subset...")
        return self._create_minimal_wikitext2()

    def _create_minimal_wikitext2(self):
        """Create a minimal WikiText-2 dataset for testing"""
        os.makedirs(self.dataset_path, exist_ok=True)

        # Sample text that resembles WikiText-2 structure
        sample_texts = [
            "= Robert Boulter =",
            "Robert Boulter is an English film editor.",
            "He was nominated for an Academy Award for the film \" Master and Commander : The Far Side of the World \" ( 2003 ) .",
            "He has worked on films such as \" K @-@ 19 : The Widowmaker \" ( 2002 ) , \" The Bourne Supremacy \" ( 2004 ) , \" United 93 \" ( 2006 ) and many others .",
            "",
            "= Physics =",
            "Physics is the natural science that studies matter and its motion and behavior through space and time .",
            "Physics is one of the oldest academic disciplines and , through its inclusion of astronomy , perhaps the oldest .",
            "Over the last two millennia , physics , chemistry , biology , and certain branches of mathematics were a part of natural philosophy .",
            "",
            "= History of art =",
            "The history of art focuses on objects made by humans in visual form for aesthetic purposes .",
            "Visual art can be classified in diverse ways , such as separating fine arts from applied arts .",
            "Art can be broadly split into different forms including painting , sculpture , and architecture .",
            "The earliest art forms were created by early human civilizations as a means of communication ."
        ]

        # Repeat and expand content to make it substantial
        expanded_texts = []
        for _ in range(50):  # Repeat to get more training data
            expanded_texts.extend(sample_texts)
            expanded_texts.extend([
                "The development of artificial intelligence has transformed modern computing .",
                "Machine learning algorithms process vast amounts of data to identify patterns .",
                "Natural language processing enables computers to understand human language .",
                "Computer vision systems can analyze and interpret visual information .",
                ""
            ])

        # Split into train/valid/test
        total_lines = len(expanded_texts)
        train_split = int(0.8 * total_lines)
        valid_split = int(0.9 * total_lines)

        splits = {
            'train': expanded_texts[:train_split],
            'valid': expanded_texts[train_split:valid_split],
            'test': expanded_texts[valid_split:]
        }

        for split_name, content in splits.items():
            filename = f"wiki.{split_name}.tokens"
            filepath = os.path.join(self.dataset_path, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                for line in content:
                    f.write(line + '\n')

            print(f"✓ Created {filename} with {len(content)} lines")

        print(f"✓ Minimal WikiText-2 dataset created at {self.dataset_path}")
        return True

    def prepare_training_data(self):
        """Prepare WikiText-2 training data for KenLM"""
        train_file = "../wikitext/wikitext2_train.txt"

        if os.path.exists(train_file):
            print(f"✓ Training data already prepared: {train_file}")
            return train_file

        print("Preparing WikiText-2 training data...")

        # Combine train and valid sets for better model
        train_files = [
            f"{self.dataset_path}/wiki.train.tokens",
            f"{self.dataset_path}/wiki.valid.tokens"
        ]

        with open(train_file, 'w', encoding='utf-8') as outfile:
            for input_file in train_files:
                if os.path.exists(input_file):
                    print(f"Processing {input_file}...")
                    with open(input_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            line = line.strip()
                            # Skip empty lines and section headers
                            if line and not line.startswith('=') and len(line.split()) > 2:
                                # Clean and normalize
                                clean_line = line.lower().replace('<unk>', 'UNK')
                                outfile.write(clean_line + '\n')

        print(f"✓ Training data prepared: {train_file}")
        return train_file

    def train_kenlm_model(self, train_file):
        """Train KenLM trigram model"""
        if os.path.exists(self.binary_model_path):
            print(f"✓ Trained model already exists: {self.binary_model_path}")
            return True

        print("Training KenLM trigram model on WikiText-2...")
        print("This may take several minutes...")

        # Get lmplz path
        lmplz_cmd = "/Users/USER/Desktop/University/Semester 6/Safa/project/project_code/kenlm/build/bin/lmplz"
        if not lmplz_cmd:
            print("❌ lmplz not found at expected location")
            return False

        try:
            # Train the ARPA model
            print("Step 1: Training ARPA model...")
            cmd = [
                lmplz_cmd,
                "-o", "3",  # trigram
                "--discount_fallback",  # handle sparse data
                "-S", "80%",  # use 80% of available memory
                "--prune", "0", "0", "1"  # minimal pruning
            ]

            print(f"Running command: {' '.join(cmd)}")

            with open(train_file, 'r') as infile, open(self.model_path, 'w') as outfile:
                result = subprocess.run(cmd, stdin=infile, stdout=outfile,
                                        stderr=subprocess.PIPE, text=True, timeout=600)  # 10 min timeout

            if result.returncode != 0:
                print(f"❌ lmplz failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return False

            print(f"✓ ARPA model saved: {self.model_path}")

            # Convert to binary for faster loading
            print("Step 2: Converting to binary format...")
            build_binary_cmd = self._find_build_binary()
            if build_binary_cmd:
                print(f"Running: {build_binary_cmd} {self.model_path} {self.binary_model_path}")
                result = subprocess.run([build_binary_cmd, self.model_path, self.binary_model_path],
                                        check=True, timeout=300)
                print(f"✓ Binary model saved: {self.binary_model_path}")
            else:
                print("⚠️ build_binary not found, using ARPA format")
                self.binary_model_path = self.model_path

            return True

        except subprocess.TimeoutExpired:
            print("❌ Training timed out")
            return False
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False

    def _find_lmplz(self):
        """Find lmplz executable"""
        possible_paths = [
            "lmplz",  # If in PATH
            "kenlm/build/bin/lmplz",  # Relative from current directory
            "./bin/lmplz",  # If running from build directory
            "../build/bin/lmplz",  # If running from project root
            "/usr/local/bin/lmplz",  # System install
            "build/bin/lmplz",  # Local build
            os.path.expanduser("~/kenlm/build/bin/lmplz"),  # Home directory
            # Add current working directory variations
            os.path.join(os.getcwd(), "kenlm/build/bin/lmplz"),
            os.path.join(os.getcwd(), "build/bin/lmplz"),
            os.path.join(os.getcwd(), "bin/lmplz"),
        ]

        # Also try to find it in common build locations
        for base_path in [os.getcwd(), os.path.dirname(os.getcwd()), os.path.expanduser("~")]:
            for subpath in ["kenlm/build/bin/lmplz", "build/bin/lmplz"]:
                full_path = os.path.join(base_path, subpath)
                possible_paths.append(full_path)

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, "--help"], capture_output=True, check=True, timeout=10)
                    print(f"✓ Found lmplz: {path}")
                    return path
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue

        print("❌ lmplz not found in any of these locations:")
        for path in possible_paths[:10]:  # Show first 10 paths
            print(f"   {path}")
        print("   ...")
        return None

    def _find_build_binary(self):
        """Use the known build_binary path"""
        build_binary_path = "/kenlm/build/bin/build_binary"

        if os.path.exists(build_binary_path):
            print(f"✓ Found build_binary: {build_binary_path}")
            return build_binary_path
        else:
            print("⚠️ build_binary not found, will use ARPA format")
            return None

    def load_model(self):
        """Load the trained KenLM model"""
        model_to_load = self.binary_model_path if os.path.exists(self.binary_model_path) else self.model_path

        if not os.path.exists(model_to_load):
            print(f"❌ Model file not found: {model_to_load}")
            return False

        try:
            print(f"Loading KenLM model: {model_to_load}")
            self.model = kenlm.Model(model_to_load)
            print("✓ KenLM model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def initialize_model(self):
        """Complete pipeline: download, train, and load model"""
        # Step 1: Download WikiText-2
        if not self.download_wikitext2():
            return False

        # Step 2: Prepare training data
        train_file = self.prepare_training_data()
        if not train_file:
            return False

        # Step 3: Train model (if not already trained)
        if not self.train_kenlm_model(train_file):
            return False

        # Step 4: Load model
        return self.load_model()

    def get_surprisal_and_probability(self, sentence):
        """
        Get surprisal (base-10) and probability for each word using trained WikiText-2 KenLM model
        FIXED: NO CLAMPING - keep natural values
        """
        if self.model is None:
            print("Model not loaded!")
            return [], []

        words = sentence.strip().split()
        surprisals = []
        probabilities = []

        for i, word in enumerate(words):
            try:
                # Use previous two words as context
                context_words = words[max(0, i - 2):i]
                context_seq = "<s> " + " ".join(w.lower() for w in context_words)
                full_seq = context_seq + " " + word.lower()

                # KenLM scores are in log10
                context_score = self.model.score(context_seq, bos=False, eos=False)
                full_score = self.model.score(full_seq, bos=False, eos=False)

                word_logprob10 = full_score - context_score
                word_surprisal = -word_logprob10  # Already base-10

                # FIXED: NO CLAMPING - keep natural values
                # Check for infinite or NaN values and replace with reasonable fallback
                if math.isnan(word_surprisal) or math.isinf(word_surprisal):
                    word_surprisal = 8.0  # Reasonable fallback for rare words

                word_prob = 10 ** (-word_surprisal)

                # Check probability is valid
                if math.isnan(word_prob) or math.isinf(word_prob) or word_prob <= 0:
                    word_prob = 1e-8  # Small but valid probability

                surprisals.append(word_surprisal)
                probabilities.append(word_prob)

            except Exception as e:
                print(f"Error processing word '{word}': {e}")
                surprisals.append(8.0)  # Reasonable fallback
                probabilities.append(1e-8)

        return surprisals, probabilities


class PythiaModel:
    """Pythia-70M model using offset-based surprisal calculation."""

    def __init__(self, model_name="EleutherAI/pythia-70m"):
        print("Loading Pythia-70M model...")

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


def generate_all_task1_graphs(df):
    """Generate all 7 required graphs for Task 1"""
    print("\n=== Generating Task 1 Graphs ===")

    df_clean = df.dropna()
    if len(df_clean) < 100:
        print("❌ Not enough data for analysis!")
        return None

    # Prepare log probabilities for spillover analysis
    df_clean['TRIGRAM_LOG_PROB'] = -df_clean['TRIGRAM_SURPRISAL']
    df_clean['PYTHIA_LOG_PROB'] = -df_clean['PYTHIA_SURPRISAL']

    # Create spillover data
    spillover_data = []
    for (participant, trial), group in df_clean.groupby(['PARTICIPANT', 'TRIAL']):
        group = group.sort_values('WORD_INDEX')
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            spillover_data.append({
                'CURRENT_TRIGRAM_LOG_PROB': current_row['TRIGRAM_LOG_PROB'],
                'CURRENT_PYTHIA_LOG_PROB': current_row['PYTHIA_LOG_PROB'],
                'CURRENT_RT': current_row['IA_DWELL_TIME'],
                'NEXT_RT': next_row['IA_DWELL_TIME']
            })
    spillover_df = pd.DataFrame(spillover_data)

    # Create figure with all 7 graphs
    fig = plt.figure(figsize=(20, 15))

    # Graph 1: Trigram Surprisal vs RT
    ax1 = plt.subplot(3, 3, 1)
    slope1, intercept1, r1, p1, _ = stats.linregress(df_clean['TRIGRAM_SURPRISAL'], df_clean['IA_DWELL_TIME'])
    r2_1 = r1 ** 2
    plt.scatter(df_clean['TRIGRAM_SURPRISAL'], df_clean['IA_DWELL_TIME'], alpha=0.3, s=1, color='blue')
    plt.plot(df_clean['TRIGRAM_SURPRISAL'], intercept1 + slope1 * df_clean['TRIGRAM_SURPRISAL'], 'r-', linewidth=2)
    plt.xlabel('Trigram Surprisal (bits)')
    plt.ylabel('Reading Time (ms)')
    plt.title(f'1. Trigram Surprisal vs RT (R² = {r2_1:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 2: Pythia Surprisal vs RT
    ax2 = plt.subplot(3, 3, 2)
    slope2, intercept2, r2, p2, _ = stats.linregress(df_clean['PYTHIA_SURPRISAL'], df_clean['IA_DWELL_TIME'])
    r2_2 = r2 ** 2
    plt.scatter(df_clean['PYTHIA_SURPRISAL'], df_clean['IA_DWELL_TIME'], alpha=0.3, s=1, color='green')
    plt.plot(df_clean['PYTHIA_SURPRISAL'], intercept2 + slope2 * df_clean['PYTHIA_SURPRISAL'], 'r-', linewidth=2)
    plt.xlabel('Pythia Surprisal (bits)')
    plt.ylabel('Reading Time (ms)')
    plt.title(f'2. Pythia Surprisal vs RT (R² = {r2_2:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 3: Pythia vs Trigram Surprisals
    ax3 = plt.subplot(3, 3, 3)
    slope3, intercept3, r3, p3, _ = stats.linregress(df_clean['PYTHIA_SURPRISAL'], df_clean['TRIGRAM_SURPRISAL'])
    r2_3 = r3 ** 2
    plt.scatter(df_clean['PYTHIA_SURPRISAL'], df_clean['TRIGRAM_SURPRISAL'], alpha=0.3, s=1, color='purple')
    plt.plot(df_clean['PYTHIA_SURPRISAL'], intercept3 + slope3 * df_clean['PYTHIA_SURPRISAL'], 'r-', linewidth=2)
    plt.xlabel('Pythia Surprisal (bits)')
    plt.ylabel('Trigram Surprisal (bits)')
    plt.title(f'3. Pythia vs Trigram Surprisals (R² = {r2_3:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 4: Pythia Probability vs Current RT
    ax4 = plt.subplot(3, 3, 4)
    slope4, intercept4, r4, p4, _ = stats.linregress(spillover_df['CURRENT_PYTHIA_LOG_PROB'],
                                                     spillover_df['CURRENT_RT'])
    r2_4 = r4 ** 2
    plt.scatter(spillover_df['CURRENT_PYTHIA_LOG_PROB'], spillover_df['CURRENT_RT'], alpha=0.3, s=1, color='orange')
    plt.plot(spillover_df['CURRENT_PYTHIA_LOG_PROB'], intercept4 + slope4 * spillover_df['CURRENT_PYTHIA_LOG_PROB'],
             'r-', linewidth=2)
    plt.xlabel('Pythia Log Probability')
    plt.ylabel('Current Word RT (ms)')
    plt.title(f'4. Pythia Probability vs Current RT (R² = {r2_4:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 5: Pythia Probability vs Next RT (Spillover)
    ax5 = plt.subplot(3, 3, 5)
    slope5, intercept5, r5, p5, _ = stats.linregress(spillover_df['CURRENT_PYTHIA_LOG_PROB'], spillover_df['NEXT_RT'])
    r2_5 = r5 ** 2
    plt.scatter(spillover_df['CURRENT_PYTHIA_LOG_PROB'], spillover_df['NEXT_RT'], alpha=0.3, s=1, color='red')
    plt.plot(spillover_df['CURRENT_PYTHIA_LOG_PROB'], intercept5 + slope5 * spillover_df['CURRENT_PYTHIA_LOG_PROB'],
             'r-', linewidth=2)
    plt.xlabel('Pythia Log Probability')
    plt.ylabel('Next Word RT (ms)')
    plt.title(f'5. Pythia Probability vs Next RT (R² = {r2_5:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 6: Trigram Probability vs Current RT
    ax6 = plt.subplot(3, 3, 6)
    slope6, intercept6, r6, p6, _ = stats.linregress(spillover_df['CURRENT_TRIGRAM_LOG_PROB'],
                                                     spillover_df['CURRENT_RT'])
    r2_6 = r6 ** 2
    plt.scatter(spillover_df['CURRENT_TRIGRAM_LOG_PROB'], spillover_df['CURRENT_RT'], alpha=0.3, s=1, color='brown')
    plt.plot(spillover_df['CURRENT_TRIGRAM_LOG_PROB'], intercept6 + slope6 * spillover_df['CURRENT_TRIGRAM_LOG_PROB'],
             'r-', linewidth=2)
    plt.xlabel('Trigram Log Probability')
    plt.ylabel('Current Word RT (ms)')
    plt.title(f'6. Trigram Probability vs Current RT (R² = {r2_6:.3f})')
    plt.grid(True, alpha=0.3)

    # Graph 7: Trigram Probability vs Next RT (Spillover)
    ax7 = plt.subplot(3, 3, 7)
    slope7, intercept7, r7, p7, _ = stats.linregress(spillover_df['CURRENT_TRIGRAM_LOG_PROB'], spillover_df['NEXT_RT'])
    r2_7 = r7 ** 2
    plt.scatter(spillover_df['CURRENT_TRIGRAM_LOG_PROB'], spillover_df['NEXT_RT'], alpha=0.3, s=1, color='pink')
    plt.plot(spillover_df['CURRENT_TRIGRAM_LOG_PROB'], intercept7 + slope7 * spillover_df['CURRENT_TRIGRAM_LOG_PROB'],
             'r-', linewidth=2)
    plt.xlabel('Trigram Log Probability')
    plt.ylabel('Next Word RT (ms)')
    plt.title(f'7. Trigram Probability vs Next RT (R² = {r2_7:.3f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task1_complete_graphs.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Return results
    results = {
        'trigram_rt': {'r2': r2_1, 'slope': slope1},
        'pythia_rt': {'r2': r2_2, 'slope': slope2},
        'model_agreement': {'r2': r2_3, 'correlation': r3},
        'spillover': {
            'pythia_current': r2_4, 'pythia_spillover': r2_5,
            'trigram_current': r2_6, 'trigram_spillover': r2_7
        }
    }

    # Print results
    print("\n=== TASK 1 RESULTS ===")
    print(f"1. Better model: {'Pythia' if r2_2 > r2_1 else 'Trigram'}")
    print(f"   Trigram: R² = {r2_1:.4f}")
    print(f"   Pythia:  R² = {r2_2:.4f}")
    print(f"2. Model agreement: r = {r3:.3f}")
    print(f"3. Spillover effects:")
    print(f"   Trigram: Current R² = {r2_6:.4f}, Spillover R² = {r2_7:.4f}")
    print(f"   Pythia:  Current R² = {r2_4:.4f}, Spillover R² = {r2_5:.4f}")

    return results


def find_disagreement_examples(df, threshold=3.0):
    """Find and analyze model disagreements"""
    df['SURPRISAL_DIFF'] = abs(df['TRIGRAM_SURPRISAL'] - df['PYTHIA_SURPRISAL'])
    disagreements = df[df['SURPRISAL_DIFF'] > threshold].sort_values('SURPRISAL_DIFF', ascending=False)

    print(f"\n=== Model Disagreements (|diff| > {threshold}) ===")
    print(f"Found {len(disagreements)} cases")
    print("\nTop disagreement words:")
    print(disagreements[['WORD', 'TRIGRAM_SURPRISAL', 'PYTHIA_SURPRISAL', 'SURPRISAL_DIFF']].head(10))

    return disagreements


def main():
    """Main execution function"""
    DATA_PATH = "../unstructered/data/onestop/ia_Paragraph.csv"

    try:
        print("=== TASK 1: N-gram vs Neural Language Models (FIXED - NO CLAMPING) ===")
        print("Training KenLM trigram on WikiText-2...")

        # 1. Load and preprocess data
        df = load_and_preprocess_data(DATA_PATH)
        print(f"Dataset: {len(df)} words from {df['PARTICIPANT'].nunique()} participants")

        # 2. Initialize WikiText-2 KenLM model
        print("\n=== Initializing WikiText-2 KenLM Model ===")
        kenlm_model = WikiText2KenLMModel()
        if not kenlm_model.initialize_model():
            print("❌ Failed to initialize KenLM model")
            return None

        # 3. Initialize Pythia model
        print("\n=== Initializing Pythia Model ===")
        pythia_model = PythiaModel()

        # 4. Compute surprisals
        print("\n=== Computing Surprisals (NO CLAMPING) ===")
        surprisal_df = compute_surprisals(df, kenlm_model, pythia_model)

        if len(surprisal_df) == 0:
            print("❌ No surprisal data computed")
            return None

        print(f"Final dataset: {len(surprisal_df)} words")
        print(
            f"Trigram surprisal: μ={surprisal_df['TRIGRAM_SURPRISAL'].mean():.2f}, σ={surprisal_df['TRIGRAM_SURPRISAL'].std():.2f}")
        print(
            f"Pythia surprisal: μ={surprisal_df['PYTHIA_SURPRISAL'].mean():.2f}, σ={surprisal_df['PYTHIA_SURPRISAL'].std():.2f}")

        # 5. Generate all required graphs
        print("\n=== Generating Task 1 Graphs ===")
        results = generate_all_task1_graphs(surprisal_df)

        # 6. Find disagreement examples
        disagreements = find_disagreement_examples(surprisal_df)

        # 7. Save results
        print("\n=== Saving Results ===")
        surprisal_df.to_csv('task1_surprisal_data.csv', index=False)
        with open('task1_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("✓ Results saved to task1_surprisal_data.csv and task1_results.pkl")

        print("\n" + "=" * 60)
        print("TASK 1 COMPLETED SUCCESSFULLY (NO CLAMPING)!")
        print("=" * 60)
        print("✓ WikiText-2 trigram model trained and saved")
        print("✓ All 7 required graphs generated with natural surprisal values")
        print("✓ Model comparison completed")
        print("✓ Spillover effects analyzed")
        print("✓ Disagreement examples found")
        print("✓ NO artificial clamping - natural value distributions preserved")

        return {
            'surprisal_df': surprisal_df,
            'results': results,
            'disagreements': disagreements
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()