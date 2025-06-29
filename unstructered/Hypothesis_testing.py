from utils import *
def main():
    """Main function to run the complete analysis with hypothesis testing"""
    print("=== PYTHIA ENTROPY ANALYSIS WITH HYPOTHESIS TESTING ===")

    # 1. Load data and model
    df = load_data()
    model, tokenizer, device = load_pythia_70m()

    # 2. Test model functionality (with FIXED surprisal calculation)
    quick_test_model(model, tokenizer, device)

    # 3. Process dataset - FULL DATASET
    print("\nProcessing dataset...")
    print("Using FULL dataset (no sampling)")

    # Process with no trial limit for full processing
    results_df = process_dataset_batch(
        df, model, tokenizer, device,
        max_trials=None,  # No limit - process all trials
        batch_size=16  # Keep batch size optimal
    )

    # 4. Validate results (but continue even if validation shows warnings)
    validation_passed = validate_results(results_df)

    if len(results_df) == 0:
        print("❌ No data to analyze, stopping")
        return None, None

    if not validation_passed:
        print("⚠️  Validation warnings detected, but continuing with available data...")

    # Print comparison to Task 1
    print(f"\n=== COMPARISON TO TASK 1 ===")
    print(f"Task 1: ~558k words processed")
    print(f"Current: {len(results_df)} words processed")
    print(f"Target: ~10k+ words for meaningful comparison")

    if len(results_df) < 5000:
        print("⚠️  Sample size much smaller than Task 1 - results may be less reliable")

    # 5. Fit regression models
    try:
        models, clean_df = fit_regression_models(results_df)

        if models and len(clean_df) > 100:  # Lower threshold but still meaningful
            # 6. Compare models
            comparison_df = compare_models(models)

            # 7. Test hypotheses
            test_hypotheses(models, clean_df)

            # 8. Create visualizations
            try:
                create_visualization(models, clean_df)
            except Exception as e:
                print(f"Visualization error (non-critical): {e}")
        else:
            print("❌ Insufficient clean data for statistical modeling")
            models = None

    except Exception as e:
        print(f"❌ Error in statistical analysis: {e}")
        models = None

    # 9. Save results
    output_file = "pythia_entropy_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")

    # 10. Final summary with Task 1 comparison
    print("\n=== FINAL SUMMARY ===")
    print(f"Total words analyzed: {len(results_df)}")
    print(f"Sample size vs Task 1: {len(results_df) / 558000 * 100:.1f}% of Task 1 size")

    if models:
        clean_count = len(clean_df) if 'clean_df' in locals() else 0
        print(f"Clean observations for modeling: {clean_count}")
        print(f"Best model by R²: Combined (R² = {models['combined']['r2']:.4f})")
        print(
            f"Entropy adds predictive power: {'Yes' if models['combined']['r2'] > models['surprisal']['r2'] else 'No'}")

        # Compare effect sizes to Task 1
        task1_surprisal_r2 = 0.026  # From your Task 1 results
        current_surprisal_r2 = models['surprisal']['r2']
        print(f"\nEffect size comparison:")
        print(f"Task 1 Surprisal R²: {task1_surprisal_r2:.4f}")
        print(f"Current Surprisal R²: {current_surprisal_r2:.4f}")
        print(f"Relative effect: {current_surprisal_r2 / task1_surprisal_r2 * 100:.1f}% of Task 1")

    else:
        print("Statistical modeling could not be completed")
        print("Recommendations:")
        print("  - Check surprisal calculation (should not be all 0.1)")
        print("  - Verify data filtering is not too strict")
        print("  - Ensure token-word alignment is working")

    return results_df, models if 'models' in locals() else None


if __name__ == "__main__":
    results, models = main()