import pandas as pd

# Example DataFrame
df = pd.DataFrame({'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})

# Example function to access row index
def func(row):
    row_index = row.name
    print(f"Row index: {row_index}")
    # Perform other computations or operations with the row data

# Apply the function to each row using df.apply(func, axis=1)
df.apply(func, axis=1)
