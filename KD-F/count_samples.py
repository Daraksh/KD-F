# import pandas as pd

# # Read CSV file
# df = pd.read_csv("/DATA2/fairune/new_val.csv")

# # Replace 'ColumnName' with the actual column you want to analyze
# col = 'FitzCategory'

# # Count samples per category
# category_counts = df[col].value_counts()

# # Number of unique categories
# num_categories = df[col].nunique()

# print("Samples per category:\n", category_counts)
# print("\nNumber of unique categories:", num_categories)

# # Optional: save counts to CSV
# category_counts.to_csv("category_counts.csv", header=['Count'])

import pandas as pd

# Read CSV file
df = pd.read_csv("/DATA2/fairune/new_test.csv")

# Filter where column 'ColumnName' has value 'target_value'
filtered_df = df[df['FitzCategory'] == 2]

# Save the filtered data to a new CSV file
filtered_df.to_csv("new_test_non_white.csv", index=False)

print(f"Saved {len(filtered_df)} rows to filtered_output.csv")
