import pandas as pd
from sklearn.utils import shuffle
df = pd.read_csv('./data/robust_double.txt', header=None)

df = shuffle(df)

df.to_csv('./data/robust_double_shuffled.csv', index=False)
