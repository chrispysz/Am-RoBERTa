import pandas as pd

train_df = pd.read_csv('./data/eval.csv', header=None)
train_df = pd.DataFrame({
    'sequence': train_df[0].replace(r'\n', ' ', regex=True),
    'label':train_df[1]
})
print(train_df.tail())

train_df.to_csv('./data/eval2.csv', index=None)