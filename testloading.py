import pandas as pd


fp = f'./data/ml-100k/u.item'
df = pd.read_csv(fp, sep='|', header=None, encoding='latin-1')
df = df.drop(columns=[0,1,2,3,4])


# Reset both row and column indices
df = df.reset_index(drop=True).T.reset_index(drop=True).T

print(
    df.to_numpy().shape
)