import pandas as pd
from sklearn.utils import shuffle


multiply_positives = True
multiply_modifier = 100
data_fold = 'f24'

def multiply_and_join(data_id_pos, data_id_neg, data_type):
    df = pd.read_csv('./data/'+data_fold+'/'+data_id_pos+'_'+data_type+'_prepared.csv', header=None)
    df_pos=df

    if multiply_positives:
        for x in range(multiply_modifier):
            df_pos = df_pos.append(df)

    df_neg = pd.read_csv('./data/'+data_fold+'/'+data_id_neg+'_'+data_type+'_prepared.csv', header=None)

    df_full = df_pos.append(df_neg)

    df_full = shuffle(df_full)

    df_full.to_csv('./data/'+data_fold+'/'+data_type+'_prepared_shuffled.csv', index=False)

multiply_and_join('hets', 'neg', 'trn')
multiply_and_join('hets', 'neg', 'val')
# multiply_and_join('hets', 'neg', 'tst')