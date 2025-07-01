#!/usr/bin/env python3
"""
 test for surprisal + entropy analysis
Concise version with mini test first, then real data processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pythia_processor import PythiaModel, load_and_preprocess_data, calculate_entropy
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def mini_test():
    """Mini test with sentences to verify calculation's work"""
    print("=== MINI TEST: Sentences ===")

    # Load model once
    pythia_model = PythiaModel()

    # Test sentences
    sentences = [
        "The cat sat on the mat.",
        "This surprising sentence contains unexpected words.",
        "Machine learning models predict surprisal values accurately."
    ]

    print("Testing surprisal (pythia_processor.py) + entropy calculations:")

    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. '{sentence}'")
        words = sentence.split()

        # Get surprisal using pythia_processor.py method
        surprisals, probs = pythia_model.get_surprisal_and_probability(sentence)

        # Calculate entropy using  method
        entropies = calculate_entropy(sentence, pythia_model.model, pythia_model.tokenizer, pythia_model.device)

        print(f"   Words: {words}")
        print(f"   Surprisals: {[f'{s:.2f}' for s in surprisals]}")
        print(f"   Entropies: {[f'{e:.2f}' for e in entropies]}")

        # Check lengths match
        if len(words) == len(surprisals) == len(entropies):
            print(f"   ✓ All lengths match ({len(words)} words)")
        else:
            print(f"   ❌ Length mismatch: {len(words)} words, {len(surprisals)} surprisals, {len(entropies)} entropies")

    print("\n✓ Mini test complete!")
    return True


def load_and_process_():
    """Load and process data using EXACT pythia_processor.py approach"""
    print("\n=== LOADING AND PROCESSING DATA ===")

    # Use EXACT same approach as pythia_processor.py
    file_path = "../unstructered/data/onestop/ia_Paragraph.csv"
    df = load_and_preprocess_data(file_path)  # This does all the preprocessing

    print(f"After pythia_processor.py preprocessing: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:")
    print(df.head(3))

    # Initialize model
    pythia_model = PythiaModel()

    results = []
    max_trials = 10078  # change to more trials, total there are 10078
    processed = 0

    print(f"\nProcessing {max_trials} trials using pythia_processor.py approach...")

    # Use EXACT same grouping and processing as pythia_processor.py
    for (participant, trial), group in df.groupby(['PARTICIPANT', 'TRIAL']):
        if processed is not None:
            if processed >= max_trials:
                break

        try:
            # EXACT same approach as pythia_processor.py
            group = group.reset_index(drop=True)  # This is key!
            words = group['WORD'].tolist()  # Now this should work
            reading_times = group['IA_DWELL_TIME'].tolist()

            print(f"Trial {trial}: {len(words)} words - {words[:5]}...")

            # Same filtering as pythia_processor.py
            if len(words) < 3 or len(words) > 100:
                continue

            # Create sentence like pythia_processor.py
            sentence = ' '.join(words)

            # Calculate surprisal using pythia_processor.py method
            surprisals, probs = pythia_model.get_surprisal_and_probability(sentence)

            # Calculate entropy
            entropies = calculate_entropy(sentence, pythia_model.model,
                                                 pythia_model.tokenizer, pythia_model.device)

            # Same alignment as pythia_processor.py
            min_len = min(len(words), len(surprisals), len(entropies), len(reading_times))

            if min_len < 3:
                continue

            # Same data storage approach as pythia_processor.py
            for i in range(min_len):
                # Same validation as pythia_processor.py
                surp_val = surprisals[i]
                ent_val = entropies[i]
                rt_val = reading_times[i]

                if (not np.isnan(surp_val) and not np.isinf(surp_val) and
                        not np.isnan(ent_val) and not np.isinf(ent_val) and
                        not np.isnan(rt_val) and not np.isinf(rt_val) and
                        surp_val > 0 and ent_val > 0 and rt_val > 0):
                    results.append({
                        'participant': participant,
                        'trial': trial,
                        'word': words[i],
                        'surprisal': surp_val,
                        'entropy': ent_val,
                        'reading_time': rt_val
                    })

            processed += 1

        except Exception as e:
            print(f"Error in trial {trial}: {e}")
            continue

    result_df = pd.DataFrame(results)
    print(f"✓ Processed {processed} trials, got {len(result_df)} word observations")

    if len(result_df) > 0:
        print(f"\nFirst few results:")
        print(result_df.head())
        print(f"\nData ranges:")
        print(f"  Surprisal: {result_df['surprisal'].min():.2f} - {result_df['surprisal'].max():.2f}")
        print(f"  Entropy: {result_df['entropy'].min():.2f} - {result_df['entropy'].max():.2f}")
        print(f"  RT: {result_df['reading_time'].min():.1f} - {result_df['reading_time'].max():.1f} ms")

    return result_df, processed


def test_hypotheses_(df):
    """Comprehensive hypothesis testing with focus on key comparisons"""
    if len(df) < 100:
        print(f"❌ Need more data for testing (have {len(df)}, need 100+)")
        return None

    print(f"\n=== COMPREHENSIVE HYPOTHESIS TESTING ({len(df)} observations) ===")

    # Prepare data
    X_surp = df[['surprisal']].values
    X_ent = df[['entropy']].values
    X_both = df[['surprisal', 'entropy']].values
    y = df['reading_time'].values

    # Fit models
    model_surp = LinearRegression().fit(X_surp, y)
    model_ent = LinearRegression().fit(X_ent, y)
    model_both = LinearRegression().fit(X_both, y)

    # Get R² values
    r2_surp = model_surp.score(X_surp, y)
    r2_ent = model_ent.score(X_ent, y)
    r2_both = model_both.score(X_both, y)

    # Statistical models for p-values and F-tests
    sm_surp = sm.OLS(y, sm.add_constant(X_surp)).fit()
    sm_ent = sm.OLS(y, sm.add_constant(X_ent)).fit()
    sm_both = sm.OLS(y, sm.add_constant(X_both)).fit()

    print(f"MODEL PERFORMANCE:")
    print(f"  1. Surprisal only: R² = {r2_surp:.4f}, p = {sm_surp.pvalues[1]:.2e}")
    print(f"  2. Entropy only:   R² = {r2_ent:.4f}, p = {sm_ent.pvalues[1]:.2e}")
    print(f"  3. Combined:       R² = {r2_both:.4f}")
    print(f"     - Surprisal p = {sm_both.pvalues[1]:.2e}")
    print(f"     - Entropy p   = {sm_both.pvalues[2]:.2e}")

    # KEY HYPOTHESIS TESTS WITH P-VALUES
    print(f"\n🧪 KEY HYPOTHESIS TESTS WITH P-VALUES:")

    # H1: Surprisal vs Entropy - Which is better predictor?
    print(f"\nH1: Surprisal vs Entropy (which predicts reading times better?)")
    print(f"   → Surprisal: R² = {r2_surp:.4f}, p = {sm_surp.pvalues[1]:.2e}")
    print(f"   → Entropy:   R² = {r2_ent:.4f}, p = {sm_ent.pvalues[1]:.2e}")

    if r2_surp > r2_ent:
        advantage = ((r2_surp - r2_ent) / r2_ent) * 100
        print(f"   → SURPRISAL WINS: {advantage:.1f}% better R² than entropy")
    else:
        advantage = ((r2_ent - r2_surp) / r2_surp) * 100
        print(f"   → ENTROPY WINS: {advantage:.1f}% better R² than surprisal")

    # Statistical significance check
    surp_significant = sm_surp.pvalues[1] < 0.05
    ent_significant = sm_ent.pvalues[1] < 0.05
    print(f"   → Surprisal significant: {'YES' if surp_significant else 'NO'} (p = {sm_surp.pvalues[1]:.2e})")
    print(f"   → Entropy significant:   {'YES' if ent_significant else 'NO'} (p = {sm_ent.pvalues[1]:.2e})")

    # H2: Does entropy add value beyond surprisal?
    print(f"\nH2: Does entropy add predictive power beyond surprisal?")
    improvement = r2_both - r2_surp
    pct_improvement = (improvement / r2_surp) * 100

    # F-test for nested models (surprisal vs surprisal+entropy)
    f_stat = ((sm_surp.ssr - sm_both.ssr) / 1) / (sm_both.ssr / sm_both.df_resid)
    f_pvalue = stats.f.sf(f_stat, 1, sm_both.df_resid)

    # Entropy coefficient significance in combined model
    entropy_coef_pvalue = sm_both.pvalues[2]
    entropy_significant_combined = entropy_coef_pvalue < 0.05

    print(f"   → R² improvement: {improvement:.4f} ({pct_improvement:.1f}% better)")
    print(f"   → F-test p-value: {f_pvalue:.2e}")
    print(f"   → Entropy coefficient p-value: {entropy_coef_pvalue:.2e}")

    if improvement > 0.001 and f_pvalue < 0.05 and entropy_significant_combined:
        print(f"   → CONCLUSION: YES - Entropy adds significant value!")
        print(f"   → Statistical evidence: F-test significant (p < 0.05)")
        print(f"   → Practical evidence: {pct_improvement:.1f}% improvement in R²")
    elif improvement > 0.001 and (f_pvalue < 0.05 or entropy_significant_combined):
        print(f"   → CONCLUSION: WEAK EVIDENCE - Some support for entropy adding value")
        f_sig = "significant" if f_pvalue < 0.05 else "not significant"
        coef_sig = "significant" if entropy_significant_combined else "not significant"
        print(f"   → F-test: {f_sig}, Entropy coefficient: {coef_sig}")
    else:
        print(f"   → CONCLUSION: NO - Entropy does not add significant value")
        print(f"   → Statistical evidence: F-test p = {f_pvalue:.3f}, Coef p = {entropy_coef_pvalue:.3f}")

    # H3: Is the combined model significantly better than each single model?
    print(f"\nH3: Statistical significance of model comparisons")

    # Combined vs Surprisal-only (already calculated above)
    print(f"   → Combined vs Surprisal-only:")
    print(f"     • F-test p-value: {f_pvalue:.2e}")
    print(f"     • Significant: {'YES' if f_pvalue < 0.05 else 'NO'}")

    # Combined vs Entropy-only
    f_stat_ent = ((sm_ent.ssr - sm_both.ssr) / 1) / (sm_both.ssr / sm_both.df_resid)
    f_pvalue_ent = stats.f.sf(f_stat_ent, 1, sm_both.df_resid)

    print(f"   → Combined vs Entropy-only:")
    print(f"     • F-test p-value: {f_pvalue_ent:.2e}")
    print(f"     • Significant: {'YES' if f_pvalue_ent < 0.05 else 'NO'}")

    # Surprisal vs Entropy direct comparison (using correlation test)
    surp_rt_corr, surp_rt_pvalue = pearsonr(df['surprisal'], df['reading_time'])
    ent_rt_corr, ent_rt_pvalue = pearsonr(df['entropy'], df['reading_time'])

    print(f"   → Direct correlation comparison:")
    print(f"     • Surprisal-RT: r = {surp_rt_corr:.3f}, p = {surp_rt_pvalue:.2e}")
    print(f"     • Entropy-RT:   r = {ent_rt_corr:.3f}, p = {ent_rt_pvalue:.2e}")

    # Effect size (Cohen's f²)
    cohens_f2 = improvement / r2_surp if r2_surp > 0 else 0
    if cohens_f2 >= 0.35:
        effect_size = "LARGE"
    elif cohens_f2 >= 0.15:
        effect_size = "MEDIUM"
    elif cohens_f2 >= 0.02:
        effect_size = "SMALL"
    else:
        effect_size = "NEGLIGIBLE"

    print(f"\nEFFECT SIZE ANALYSIS:")
    print(f"   → Cohen's f² = {cohens_f2:.3f} ({effect_size} effect)")
    print(f"   → Interpretation:")
    if effect_size == "LARGE":
        print(f"     • Large practical significance - entropy substantially improves prediction")
    elif effect_size == "MEDIUM":
        print(f"     • Medium practical significance - entropy meaningfully improves prediction")
    elif effect_size == "SMALL":
        print(f"     • Small practical significance - entropy slightly improves prediction")
    else:
        print(f"     • Negligible practical significance - entropy barely improves prediction")

    # Overall summary with p-values
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"   → Best single predictor: {'Surprisal' if r2_surp > r2_ent else 'Entropy'}")
    print(f"   → Entropy adds significant value: {'YES' if f_pvalue < 0.05 and entropy_significant_combined else 'NO'}")
    print(f"   → Key p-values:")
    print(f"     • Surprisal significance: p = {sm_surp.pvalues[1]:.2e}")
    print(f"     • Entropy significance: p = {sm_ent.pvalues[1]:.2e}")
    print(f"     • Entropy in combined model: p = {entropy_coef_pvalue:.2e}")
    print(f"     • Model improvement F-test: p = {f_pvalue:.2e}")

    return {
        'r2_surp': r2_surp,
        'r2_ent': r2_ent,
        'r2_both': r2_both,
        'improvement': improvement,
        'f_pvalue': f_pvalue,
        'entropy_pvalue': entropy_coef_pvalue,
        'surprisal_pvalue': sm_surp.pvalues[1],
        'entropy_single_pvalue': sm_ent.pvalues[1],
        'effect_size': effect_size,
        'cohens_f2': cohens_f2
    }


def create__plot(df):
    """Create  visualization"""
    if len(df) < 10:
        print("Not enough data for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Surprisal vs RT
    axes[0].scatter(df['surprisal'], df['reading_time'], alpha=0.5, s=10)
    axes[0].set_xlabel('Surprisal')
    axes[0].set_ylabel('Reading Time (ms)')
    axes[0].set_title(f'Surprisal vs RT (r={df["surprisal"].corr(df["reading_time"]):.3f})')

    # Entropy vs RT
    axes[1].scatter(df['entropy'], df['reading_time'], alpha=0.5, s=10, color='orange')
    axes[1].set_xlabel('Entropy')
    axes[1].set_ylabel('Reading Time (ms)')
    axes[1].set_title(f'Entropy vs RT (r={df["entropy"].corr(df["reading_time"]):.3f})')

    # Surprisal vs Entropy
    axes[2].scatter(df['surprisal'], df['entropy'], alpha=0.5, s=10, color='green')
    axes[2].set_xlabel('Surprisal')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title(f'Surprisal vs Entropy (r={df["surprisal"].corr(df["entropy"]):.3f})')
    # plotting
    plt.tight_layout()
    plt.savefig('entropy_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Plot saved as '_entropy_analysis.png'")
