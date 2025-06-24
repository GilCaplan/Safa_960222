from utils import *

model, tokenizer, device = load_pythia_70m()

data = load_data()
results_df = process_dataset_surprisal_entropy(data, model, tokenizer, device)
validate_results(results_df)
