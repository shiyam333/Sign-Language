import pandas as pd

# 1. Load CSV
df = pd.read_csv("../data/asl_landmarks.csv")

# 2. Basic info
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

# 3. Check label distribution (assuming column name like 'label' or 'class')
print(df['label'].value_counts())
