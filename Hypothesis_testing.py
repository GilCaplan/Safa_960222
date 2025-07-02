from utils import *
def main():
    """Main function - concise test"""
    print("=== SURPRISAL + ENTROPY TEST ===")

    # 1. Mini test first
    if not mini_test():
        print("❌ Mini test failed")
        return

    # 2. Process real data
    df, processed_trials = load_and_process_()

    if len(df) == 0:
        print("❌ No data processed successfully")
        return

    # 3. Test hypotheses
    results = test_hypotheses_(df)

    # 4. Create plot
    create__plot(df)

    # 5. Save results
    df.to_csv('_test_results.csv', index=False)
    print(f"\n✓ Results saved to '_test_results.csv'")

    # 6. Summary
    print(f"\n=== FINAL SUMMARY ===")
    if results:
        print(f"✅ SUCCESS: Processed {len(df)} word observations from {processed_trials} trials")
        print(f"   📊 Model Performance:")
        print(f"      • Surprisal only: R² = {results['r2_surp']:.4f}")
        print(f"      • Entropy only:   R² = {results['r2_ent']:.4f}")
        print(f"      • Combined:       R² = {results['r2_both']:.4f}")

        # Key findings
        improvement = results['r2_both'] - results['r2_surp']
        pct_improvement = (improvement / results['r2_surp']) * 100

        print(f"\n   🔍 KEY FINDINGS:")
        if results['r2_surp'] > results['r2_ent']:
            print(f"      • Surprisal is better single predictor than entropy")
        else:
            print(f"      • Entropy is better single predictor than surprisal")

        if improvement > 0.001 and results.get('f_pvalue', 1) < 0.05:
            print(f"      • ✅ Entropy DOES add significant predictive power!")
            print(f"      • 📈 Improvement: +{improvement:.4f} R² ({pct_improvement:.1f}% better)")
            print(f"      • 📏 Effect size: {results.get('effect_size', 'unknown').lower()}")
        else:
            print(f"      • ❌ Entropy does NOT add significant predictive power")

        # Compare to pythia_processor.py
        task1_pythia_r2 = 0.013  # From pythia_processor.py results
        current_surprisal_r2 = results['r2_surp']

        print(f"\n   📋 COMPARISON TO TASK 1:")
        print(f"      • Task 1 Pythia R²: {task1_pythia_r2:.4f}")
        print(f"      • Our Surprisal R²: {current_surprisal_r2:.4f}")
        print(f"      • Relative effect: {current_surprisal_r2 / task1_pythia_r2:.1f}x stronger")

        if results['r2_both'] > current_surprisal_r2:
            added_variance = (results['r2_both'] - current_surprisal_r2) * 100
            print(f"      • 🆕 Entropy adds {added_variance:.2f}% more explained variance!")

    else:
        print(f"⚠️ Partial success: Data processed but insufficient for modeling")
        print(f"   • Processed {len(df)} observations")
        print(f"   • Try increasing max_trials for more data")

    return df


if __name__ == "__main__":
    results = main()