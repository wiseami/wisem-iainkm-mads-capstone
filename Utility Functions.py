import pandas as pd
import numpy as np
import os


def dedupe(file, column):
    df = pd.read_csv(file)
    og_len = len(df)
    dupes = og_len - len(df.drop_duplicates(subset=[column], keep='first'))
    df.drop_duplicates(subset=[column], keep='first', inplace=True, ignore_index=True)
    df.to_csv(file, index=False)
    print('Dedupe complete. {} duplicates removed.'.format(dupes))


def combine_csv(file_1, file_2, column, new_file):
    df_1 = pd.read_csv(file_1)
    df_2 = pd.read_csv(file_2)
    for d in [df_1, df_2]:
        if 'id' in d.columns.to_list():
            d.rename(columns={'id':'track_id'}, inplace=True)
    df_f = df_1.merge(df_2, how='left', on=[column])
    df_f.to_csv(new_file)



combine_csv('lookups/track_audio_features.csv', 'lookups/MIR_features.csv', 'track_id', 'lookups/full_audio_features')

# dedupe('lookups/MIR_features.csv', 'track_id')


