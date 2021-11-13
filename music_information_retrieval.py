# The purpose of this script is to extract additional features from track previews for
# future analysis and modelling

import pandas as pd
import requests
import librosa
from tqdm import tqdm
import os
from collections import defaultdict



def load_tracks(list_file_name):
    df = pd.read_csv(list_file_name)
    df = df[df['track_preview_url'].notna()]
    return df


def extract_mp3(url, save_folder, file_name):
    doc = requests.get(url)
    with open(save_folder+file_name, 'wb') as f:
        f.write(doc.content)


def delete_mp3(file):
    os.remove(file)


# get features from librosa
def get_features(file):
    y, sr = librosa.load(file)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    roll = librosa.feature.spectral_rolloff(y).mean()
    onset = librosa.onset.onset_strength(y).mean()
    output = {'ZCR': [zcr],
              'Spectral_Rolloff': [roll],
              'Onset_Strength': [onset]
              }
    return output


new = None
end_df = None
df_feat = None


def process_tracks(source_file, end_file, max_rows=0):
    global new, end_df, df_feat
    features_dict = defaultdict(list)
    source_df = load_tracks(source_file)

    if max_rows > 0:
        source_df = source_df.head(max_rows)
    else:
        source_df = source_df

    if not os.path.isfile(end_file):
        # global new
        print('{} does not exist. Creating new file from scratch.'.format(end_file))
        new = True
        tracks = source_df
    else:
        # global end_df
        end_df = pd.read_csv(end_file)
        tracks = source_df[~source_df['track_id'].isin(end_df['track_id'])]

    if len(tracks) > 0:
        # global df_feat
        total = len(tracks)
        for num, tup in enumerate(tracks.iterrows()):
            idx, row = tup
            print('Processing track {} of {}.'.format(num+1, total))
            track_id = row.track_id
            url = row.track_preview_url
            audio_folder = 'audio_files/'
            file_name = str(track_id)+'.mp3'
            audio_file = audio_folder+file_name

            extract_mp3(url, audio_folder, file_name)

            temp_features = get_features(audio_file)

            features_dict['track_id'].append(track_id)
            for key, value in temp_features.items():
                features_dict[key].append(value)

            delete_mp3(audio_file)
        df_feat = pd.DataFrame(features_dict)

        if new:
            df_feat.to_csv(end_file, index=False)
            return df_feat
        else:
            end_df = end_df.append(df_feat, ignore_index=True)
            end_df.to_csv(end_file, index=False)
            return end_df
    else:
        print('No new tracks to process.')


source_file = 'playlist_data/playlist_data.csv'
end_file = 'lookups/MIR_features.csv'

print(process_tracks(source_file, end_file, max_rows=100))




