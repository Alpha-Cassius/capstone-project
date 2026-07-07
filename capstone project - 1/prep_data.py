import pandas as pd
import numpy as np

# Download Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Make it messy to fit all assignment criteria perfectly:

# 1. Incorrect inferred dtype: Prepend '$' to Fare so it becomes an object
df['Fare'] = df['Fare'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else x)

# 2. Introduce some nulls into Fare and SibSp so Task 9a works well
np.random.seed(42)
fare_null_idx = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
df.loc[fare_null_idx, 'Fare'] = np.nan

sibsp_null_idx = np.random.choice(df.index, size=int(len(df)*0.1), replace=False)
df.loc[sibsp_null_idx, 'SibSp'] = np.nan

# 3. Add duplicates: sample 15 rows and append them
duplicates = df.sample(15, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# 4. Save as raw_client_data.csv
df.to_csv("raw_client_data.csv", index=False)
print("raw_client_data.csv recreated with nulls in highly skewed columns. Shape:", df.shape)
