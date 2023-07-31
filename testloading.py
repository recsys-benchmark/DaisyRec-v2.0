import pandas as pd


fp = f'./data/ml-100k/u.item'
df = pd.read_csv(fp, sep='|', header=None, encoding='latin-1')

print(
    df.head(10)
)