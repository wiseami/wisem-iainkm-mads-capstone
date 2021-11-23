# The purpose of this script is to extract additional features from track previews for
# future analysis and modelling

import pandas as pd
import requests
import librosa
from tqdm import tqdm
import os
from collections import defaultdict
import numpy as np



def load_tracks(list_file_name):
    df = pd.read_csv(list_file_name)
    df = df[df['preview_url'].notna()]
    df.rename(columns={'id': 'track_id'}, inplace=True)
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
    S = np.abs(librosa.stft(y)) #spectral magnitude
    onset_env = librosa.onset.onset_strength(y)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).mean()
    mff = librosa.feature.mfcc(y=y, sr=sr).mean()
    spec_cen = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr).mean()
    spec_flat = librosa.feature.spectral_flatness(y=y).mean()
    roll = librosa.feature.spectral_rolloff(y).mean()
    poly = librosa.feature.poly_features(S=S, order=1).mean()
    ton = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    onset = onset_env.mean()
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch = pitch.mean()
    mag = mag.mean()
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)


    output = {
              'chroma': [chroma],
              'chroma_cens': [chroma_cens],
              'mff': [mff],
              'spectral_centroid': [spec_cen],
              'spectral_bandwidth': [spec_band],
              'spectral_contrast': [spec_cont],
              'spectral_flatness': [spec_flat],
              'Spectral_Rolloff': [roll],
              'poly_features': [poly],
              'tonnetz': [ton],
              'ZCR': [zcr],
              'onset_strength': [onset],
              'pitch': [pitch],
              'magnitude': [mag],
              'tempo': [tempo]
              }
    return output


new = None
end_df = None
df_feat = None


def process_tracks(source_file, end_file, max_rows=0):
    global new, end_df, df_feat
    features_dict = defaultdict(list)
    source_df = load_tracks(source_file)

    # handle limiting the number of rows to process for testing purposes
    if max_rows > 0:
        source_df = source_df.head(max_rows)
    else:
        source_df = source_df

    # handle creating a file if it doesn't already exist
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
        print('{} tracks to process'.format(total))
        batchsize = 50
        for i in range(0, total, batchsize):
            features_dict = defaultdict(list)
            if i+batchsize > total:
                batch = tracks.iloc[i: -1]
            else:
                batch = tracks.iloc[i: i+batchsize]
            for num, tup in enumerate(batch.iterrows()):
                idx, row = tup
                # print('Processing track {} of {}.'.format(num+1, total))
                track_id = row.track_id
                url = row.preview_url
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

            if not os.path.isfile(end_file):
                df_feat.to_csv(end_file, index=False)
                # return df_feat
            else:
                end_df = end_df.append(df_feat, ignore_index=True)
                end_df.to_csv(end_file, index=False)
                # return end_df
            print('Tracks {} through {} finished.'.format(i, i+batchsize))
    else:
        print('No new tracks to process.')
    print('Feature extraction function complete.')


source_file = 'lookups/track_audio_features.csv'
end_file = 'lookups/MIR_features.csv'

print(process_tracks(source_file, end_file))




