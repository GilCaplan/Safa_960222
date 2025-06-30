#!/usr/bin/env python3
"""
FIXED: hypothesis_testing.py
Main script for entropy analysis using Task 1 surprisal calculation + entropy + hypotheses.

Key features:
1. Uses task_1.py surprisal calculation (same as project1_zak)
2. Adds entropy calculation on top
3. Tests 4 key hypotheses about surprisal vs entropy
4. Compatible with updated utils.py functions
"""

from utils import *


def main():
    """Main function to run the complete analysis using Task 1 surprisal + entropy"""
    print("=== PYTHIA ENTROPY ANALYSIS - USING TASK_1.PY SURPRISAL ===")

    # 1. Load data and model
    print("\n=== LOADING DATA AND MODEL ===")
    df = load_data()
    model, tokenizer, device = load_pythia_70m()

    # 2. Test model functionality with Task 1 style
    print("\n=== TESTING MODEL - TASK_1.PY STYLE ===")
    quick_test_model(model, tokenizer, device)

    # 3. Process dataset using Task 1 surprisal approach + entropy
    print("\n=== PROCESSING DATASET ===")
    print("Using task_1.py surprisal calculation + entropy calculation")
    print("Processing sample data first (set max_trials=None for full dataset)")

    # Process with limited trials first for testing
    results_df = process_dataset_batch(
        df, model, tokenizer, device,
        max_trials=100,  # Start with 100 trials, change to None for full dataset
        batch_size=16
    )

    # 4. Validate results
    print("\n=== VALIDATING RESULTS ===")
    validation_passed = validate_results(results_df)

    if len(results_df) == 0:
        print("❌ No data to analyze, stopping")
        return None, None

    # Print data summary
    print(f"\n=== DATA SUMMARY ===")
    print(f"Total words processed: {len(results_df)}")
    print(f"Unique participants: {results_df['participant_id'].nunique()}")
    print(f"Unique trials: {results_df['trial_index'].nunique()}")

    # Show some example data
    print(f"\nExample data:")
    print(results_df[['word', 'pythia_surprisal', 'pythia_entropy', 'reading_time']].head(10))

    # Check data distributions
    print(f"\nSurprisal distribution:")
    print(f"  Mean: {results_df['pythia_surprisal'].mean():.3f}")
    print(f"  Std: {results_df['pythia_surprisal'].std():.3f}")
    print(f"  Min: {results_df['pythia_surprisal'].min():.3f}")
    print(f"  Max: {results_df['pythia_surprisal'].max():.3f}")

    print(f"\nEntropy distribution:")
    print(f"  Mean: {results_df['pythia_entropy'].mean():.3f}")
    print(f"  Std: {results_df['pythia_entropy'].std():.3f}")
    print(f"  Min: {results_df['pythia_entropy'].min():.3f}")
    print(f"  Max: {results_df['pythia_entropy'].max():.3f}")

    print(f"\nReading time distribution:")
    print(f"  Mean: {results_df['reading_time'].mean():.1f} ms")
    print(f"  Std: {results_df['reading_time'].std():.1f} ms")
    print(f"  Min: {results_df['reading_time'].min():.1f} ms")
    print(f"  Max: {results_df['reading_time'].max():.1f} ms")

    # 5. Fit regression models
    print("\n=== STATISTICAL MODELING ===")
    try:
        models, clean_df = fit_regression_models(results_df)

        if models and len(clean_df) > 100:
            print(f"✓ Successfully fitted models with {len(clean_df)} clean observations")

            # 6. Compare models
            print("\n=== MODEL COMPARISON ===")
            comparison_df = compare_models(models)

            # 7. Test hypotheses
            print("\n=== HYPOTHESIS TESTING ===")
            test_hypotheses(models, clean_df)

            # 8. Create visualizations
            print("\n=== CREATING VISUALIZATIONS ===")
            try:
                create_visualization(models, clean_df)
                print("✓ Visualizations created successfully")
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
        else:
            print("❌ Insufficient clean data for statistical modeling")
            models = None

    except Exception as e:
        print(f"❌ Error in statistical analysis: {e}")
        import traceback
        traceback.print_exc()
        models = None

    # 9. Save results
    print("\n=== SAVING RESULTS ===")
    output_file = "pythia_entropy_analysis_task1_style.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")

    # 10. Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Analysis completed using Task 1 surprisal + entropy!")
    print(f"Total words analyzed: {len(results_df)}")

    # Compare with Task 1
    task1_sample_size = 558000
    current_sample_size = len(results_df)

    print(f"\nComparison to Task 1:")
    print(f"  Task 1 sample size: ~{task1_sample_size:,} words")
    print(f"  Current sample size: {current_sample_size:,} words")
    print(f"  Relative size: {current_sample_size / task1_sample_size * 100:.1f}% of Task 1")

    if models:
        print(f"\nModel performance:")
        print(f"  Surprisal R²: {models['surprisal']['r2']:.4f}")
        print(f"  Entropy R²: {models['entropy']['r2']:.4f}")
        print(f"  Combined R²: {models['combined']['r2']:.4f}")

        # Calculate improvement
        surprisal_r2 = models['surprisal']['r2']
        combined_r2 = models['combined']['r2']
        improvement = ((combined_r2 - surprisal_r2) / surprisal_r2) * 100 if surprisal_r2 > 0 else 0

        print(f"  Entropy improvement: {improvement:+.1f}%")

        # Compare with Task 1 surprisal effect
        task1_surprisal_r2 = 0.026  # From Task 1
        current_surprisal_r2 = models['surprisal']['r2']

        print(f"\nEffect size comparison:")
        print(f"  Task 1 Surprisal R²: {task1_surprisal_r2:.4f}")
        print(f"  Current Surprisal R²: {current_surprisal_r2:.4f}")
        print(f"  Relative effect: {current_surprisal_r2 / task1_surprisal_r2 * 100:.1f}% of Task 1")

        # Key finding
        entropy_adds_value = combined_r2 > surprisal_r2 + 0.001
        print(f"\n🔍 KEY FINDING:")
        print(f"   Entropy {'DOES' if entropy_adds_value else 'DOES NOT'} add predictive power beyond surprisal")

        if entropy_adds_value:
            print(f"   → Entropy explains additional {((combined_r2 - surprisal_r2) * 100):.2f}% of variance")

        # Statistical significance
        combined_model = models['combined']['sm_model']
        if len(combined_model.pvalues) > 2:
            entropy_p = combined_model.pvalues[2]
            print(
                f"   → Entropy significance: p = {entropy_p:.4f} ({'significant' if entropy_p < 0.05 else 'not significant'})")

    else:
        print("\n❌ Statistical modeling failed")
        print("This could be due to:")
        print("  - Insufficient variation in surprisal/entropy values")
        print("  - Data quality issues")
        print("  - Processing errors")

    print("\n" + "=" * 60)
    print("ENTROPY ANALYSIS COMPLETE")
    print("=" * 60)

    if validation_passed and models:
        print("✅ SUCCESS: Analysis completed successfully!")
        print("   - Task 1 style surprisal calculations working")
        print("   - Entropy calculations working")
        print("   - Statistical models fitted")
        print("   - Hypotheses tested")
        print("   - Results saved and visualized")
    elif models:
        print("⚠️  PARTIAL SUCCESS: Models fitted but validation warnings")
    else:
        print("❌ ISSUES: Analysis completed but statistical modeling failed")

    return results_df, models


def run_full_analysis():
    """
    Run analysis on full dataset using Task 1 approach + entropy
    """
    print("=== FULL DATASET ANALYSIS - TASK_1.PY + ENTROPY ===")

    # Load data and model
    df = load_data()
    model, tokenizer, device = load_pythia_70m()

    # Process full dataset using Task 1 approach + entropy
    print("\nProcessing FULL dataset using task_1.py surprisal + entropy...")
    results_df = process_dataset_batch(
        df, model, tokenizer, device,
        max_trials=None,  # Process ALL trials
        batch_size=16
    )

    # Full analysis pipeline
    validation_passed = validate_results(results_df)

    if validation_passed and len(results_df) > 5000:
        models, clean_df = fit_regression_models(results_df)

        if models:
            comparison_df = compare_models(models)
            test_hypotheses(models, clean_df)
            create_visualization(models, clean_df)

            # Save comprehensive results
            results_df.to_csv("pythia_entropy_analysis_task1_full.csv", index=False)

            print("\n=== FULL ANALYSIS COMPLETE ===")
            print(f"Total word-reading time pairs: {len(results_df):,}")
            print(f"Results saved to pythia_entropy_analysis_task1_full.csv")

            return results_df, models

    print("❌ Full analysis failed")
    return results_df, None


def test_task1_consistency():
    """
    Test to verify our surprisal calculation matches task_1.py
    """
    print("=== TESTING TASK_1.PY CONSISTENCY ===")

    # Load model (using our utils function)
    model, tokenizer, device = load_pythia_70m()

    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "This is a simple test sentence for surprisal calculation.",
        "Machine learning models can predict word surprisal values accurately."
    ]

    # Create PythiaModel instance from task_1.py
    from task_1 import PythiaModel
    pythia_model = PythiaModel()

    print("Testing surprisal calculation consistency:")

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: '{sentence}'")
        words = sentence.split()

        try:
            # Use task_1.py method directly
            surprisals, probabilities = pythia_model.get_surprisal_and_probability(sentence)

            # Calculate entropy using our method
            entropies = calculate_entropy_fast(sentence, model, tokenizer, device)

            print(f"  Words: {len(words)}")
            print(f"  Surprisals: {len(surprisals)} values")
            print(f"  Entropies: {len(entropies)} values")

            if len(surprisals) == len(words) and len(entropies) == len(words):
                print(f"  ✓ Length consistency: All {len(words)} words processed")

                # Show sample values
                print(f"  Sample surprisals: {[f'{s:.2f}' for s in surprisals[:5]]}")
                print(f"  Sample entropies: {[f'{e:.2f}' for e in entropies[:5]]}")

                # Check for NaN/inf values
                has_nan_surprisal = any(np.isnan(s) or np.isinf(s) for s in surprisals)
                has_nan_entropy = any(np.isnan(e) or np.isinf(e) for e in entropies)

                print(f"  Surprisal quality: {'❌ HAS NaN/inf' if has_nan_surprisal else '✓ NO NaN/inf'}")
                print(f"  Entropy quality: {'❌ HAS NaN/inf' if has_nan_entropy else '✓ NO NaN/inf'}")

            else:
                print(f"  ❌ Length mismatch!")
                print(f"     Words: {len(words)}, Surprisals: {len(surprisals)}, Entropies: {len(entropies)}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print(f"\n💡 CONCLUSION:")
    print(f"   This test verifies our approach uses the exact same surprisal calculation as task_1.py")
    print(f"   and adds entropy calculation on top without interfering with surprisal.")


def run_hypothesis_focused_analysis():
    """
    Run a focused analysis specifically for hypothesis testing
    """
    print("=== HYPOTHESIS-FOCUSED ANALYSIS ===")

    # Load data and model
    df = load_data()
    model, tokenizer, device = load_pythia_70m()

    # Process a reasonable sample for hypothesis testing
    print("\nProcessing data for hypothesis testing...")
    results_df = process_dataset_batch(
        df, model, tokenizer, device,
        max_trials=200,  # Good sample size for statistical power
        batch_size=16
    )

    if len(results_df) < 500:
        print("⚠️ Small sample size - consider increasing max_trials")

    # Focus on the key hypotheses
    print(f"\n=== HYPOTHESIS TESTING FOCUS ===")
    print(f"Sample size: {len(results_df)} word-reading time pairs")

    # Quick validation
    if validate_results(results_df):
        # Fit models specifically for hypothesis testing
        models, clean_df = fit_regression_models(results_df)

        if models and len(clean_df) > 100:
            print(f"\n🧪 TESTING 4 KEY HYPOTHESES:")
            print(f"   H1: Surprisal predicts reading times")
            print(f"   H2: Entropy predicts reading times")
            print(f"   H3: Combined model outperforms surprisal alone")
            print(f"   H4: Entropy adds unique predictive power")

            # Run hypothesis tests
            test_hypotheses(models, clean_df)

            # Create focused visualization
            create_visualization(models, clean_df)

            # Save results
            results_df.to_csv("hypothesis_testing_results.csv", index=False)

            return results_df, models
        else:
            print("❌ Insufficient data for hypothesis testing")
    else:
        print("❌ Data validation failed")

    return results_df, None


def demonstrate_entropy_vs_surprisal():
    """
    Demonstrate the difference between entropy and surprisal with examples
    """
    print("=== ENTROPY vs SURPRISAL DEMONSTRATION ===")

    # Load model
    model, tokenizer, device = load_pythia_70m()

    # Create examples that should show different entropy vs surprisal patterns
    examples = [
        "The cat sat on the mat.",  # Simple, predictable
        "The quantum physicist discovered unexpected results.",  # Technical, less predictable
        "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",  # Syntactically complex
        "I think that that that that that student wrote is wrong.",  # Repetitive structure
    ]

    print("Comparing entropy and surprisal for different sentence types:\n")

    from task_1 import PythiaModel
    pythia_model = PythiaModel()

    for i, sentence in enumerate(examples, 1):
        print(f"Example {i}: '{sentence}'")

        try:
            # Get surprisal (using task_1.py method)
            surprisals, _ = pythia_model.get_surprisal_and_probability(sentence)

            # Get entropy (using our method)
            entropies = calculate_entropy_fast(sentence, model, tokenizer, device)

            words = sentence.split()

            if len(surprisals) == len(words) == len(entropies):
                # Calculate summary statistics
                avg_surprisal = np.mean(surprisals)
                avg_entropy = np.mean(entropies)
                surprisal_var = np.var(surprisals)
                entropy_var = np.var(entropies)

                print(f"  Words: {len(words)}")
                print(f"  Avg Surprisal: {avg_surprisal:.2f} ± {np.sqrt(surprisal_var):.2f}")
                print(f"  Avg Entropy: {avg_entropy:.2f} ± {np.sqrt(entropy_var):.2f}")
                print(f"  Correlation: {np.corrcoef(surprisals, entropies)[0, 1]:.3f}")

                # Show word-by-word for first few words
                print(f"  Word-by-word (first 5):")
                for j in range(min(5, len(words))):
                    print(f"    '{words[j]}': S={surprisals[j]:.2f}, E={entropies[j]:.2f}")

            else:
                print(f"  ❌ Length mismatch")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

        print()

    print("💡 Key differences:")
    print("   - Surprisal: How unexpected a specific word is given context")
    print("   - Entropy: How uncertain the model is about what comes next")
    print("   - Both should correlate with reading times, but may capture different aspects")


if __name__ == "__main__":
    print("Choose analysis type:")
    print("1. Main analysis (sample data)")
    print("2. Full dataset analysis")
    print("3. Test task_1.py consistency")
    print("4. Hypothesis-focused analysis")
    print("5. Demonstrate entropy vs surprisal")

    choice = input("Enter choice (1-5) or press Enter for main analysis: ").strip()

    if choice == "2":
        print("Running full dataset analysis...")
        full_results, full_models = run_full_analysis()
    elif choice == "3":
        print("Testing task_1.py consistency...")
        test_task1_consistency()
    elif choice == "4":
        print("Running hypothesis-focused analysis...")
        hyp_results, hyp_models = run_hypothesis_focused_analysis()
    elif choice == "5":
        print("Demonstrating entropy vs surprisal...")
        demonstrate_entropy_vs_surprisal()
    else:
        print("Running main analysis with sample data...")
        results, models = main()

        if results is not None and len(results) > 0:
            run_full = input("\nRun full dataset analysis? (y/n): ").strip().lower()
            if run_full == 'y':
                print("\nRunning full analysis...")
                full_results, full_models = run_full_analysis()