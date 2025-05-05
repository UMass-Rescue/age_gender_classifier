import pandas as pd

# Load the CSV
df = pd.read_csv('temp_output_labeled.csv')

# Extract true_label (only need one per imageId, they're identical across the three models)
true_labels = df[['imageId', 'true_label']].drop_duplicates()

# Pivot the data: get label and confidence per model
pivot_df = df.pivot_table(
    index='imageId',
    columns='model_name',
    values=['label', 'confidence'],
    aggfunc='first'
)

# Flatten the multi-level column names
pivot_df.columns = [f'{model}_{field}' for field, model in pivot_df.columns]

# Merge the true_label back in
final_df = pivot_df.reset_index().merge(true_labels, on='imageId')

# Save to CSV
final_df.to_csv('pivoted_output.csv', index=False)

print("Done! Output saved to 'pivoted_output_with_true_label.csv'")
