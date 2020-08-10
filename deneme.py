import pandas as pd

df = pd.read_csv("real.csv")

print(df.label.value_counts())