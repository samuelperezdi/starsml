import pandas as pd

df = pd.read_parquet('full_results_with_probabilities.parquet')
df.to_csv('full_results_with_probabilities.csv', index=False)