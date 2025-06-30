from utils import *


def main():
    """Main function to run entropy analysis with hypothesis testing"""
    print("=== PYTHIA ENTROPY ANALYSIS (UNSTRUCTURED PART) ===")
    print("Word-by-word processing like Task 1")

    # 1. Load data and model
    try:
        df = load_data()
        model, tokenizer, device = load_pythia_70m()
        quick_test_model(model, tokenizer, device)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None, None

    # 2. Process dataset word-by-word
    print("\nProcessing dataset word-by-word...")
    try:
        # Start with test sample
        results_df = process_dataset(df, model, tokenizer, device, max_trials=1000)

        # Scale up if test works
        if len(results_df) > 5000:
            print("✓ Test successful, processing full dataset...")
            results_df = process_dataset(df, model, tokenizer, device, max_trials=None)

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None, None

    # 3. Validate and analyze
    validate_results(results_df)

    if len(results_df) < 1000:
        print("❌ Insufficient data for analysis")
        return None, None

    print(f"\n✓ Processed {len(results_df)} words ({len(results_df) / 558000 * 100:.1f}% of Task 1 size)")

    # 4. Statistical modeling
    try:
        models, clean_df = fit_regression_models(results_df)

        if models:
            compare_models(models)
            test_hypotheses(models, clean_df)
            create_visualization(models, clean_df)

            # Save results
            results_df.to_csv("pythia_entropy_analysis.csv", index=False)
            print("✓ Results saved to pythia_entropy_analysis.csv")

            # Summary
            print(f"\n=== SUMMARY ===")
            print(f"Surprisal R²: {models['surprisal']['r2']:.4f} (Task 1: ~0.026)")
            print(f"Combined R²: {models['combined']['r2']:.4f}")
            print(f"Entropy adds: {models['combined']['r2'] - models['surprisal']['r2']:.4f}")

            entropy_p = models['combined']['sm_model'].pvalues[2]
            if entropy_p < 0.05:
                print("✓ Entropy significantly improves prediction")
            else:
                print("❌ Entropy does not significantly improve prediction")

        else:
            print("❌ Statistical modeling failed")
            models = None

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        models = None

    return results_df, models


def extended_analysis(results_df):
    """Brief extended analysis"""
    if len(results_df) < 1000:
        return

    print("\n=== EXTENDED ANALYSIS ===")

    # Word position effects
    if 'word_position' in results_df.columns:
        pos_stats = results_df.groupby('word_position').agg({
            'reading_time': 'mean', 'pythia_surprisal': 'mean'
        }).round(2)
        print("First 5 positions:")
        print(pos_stats.head())

    # Word length effects
    results_df['word_length'] = results_df['word'].str.len()
    len_stats = results_df.groupby('word_length').agg({
        'reading_time': 'mean', 'pythia_surprisal': 'mean'
    }).round(2)
    print("\nWord length effects:")
    print(len_stats.head())


if __name__ == "__main__":
    print("Project 1 - Unstructured Part: Entropy Analysis")
    print("=" * 50)

    # Run analysis
    results, models = main()

    # Extended analysis if successful
    if results is not None and len(results) > 1000:
        extended_analysis(results)

    print("\n" + "=" * 50)
    if models:
        print("✓ Analysis complete - check results and visualizations")
    else:
        print("❌ Analysis failed - check setup and data")