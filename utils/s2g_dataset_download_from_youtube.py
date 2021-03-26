from __future__ import unicode_literals

import argparse
from subprocess import call

import cv2
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', default='/media/uttaran/repo1/data/s2g',
                    help='base folder path of dataset')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker ')
args = parser.parse_args()

speakers = ['almaram', 'angelica', 'chemistry', 'conan', 'ellen', 'jon', 'oliver', 'rock,', 'seth', 'shelly']
BASE_PATH = args.base_path
df = pd.read_csv(os.path.join(BASE_PATH, 'videos_links.csv'))

temp_output_path = os.path.join(BASE_PATH, 'tmp/temp_video.mp4')

for speaker in speakers:
    df_by_speaker = df[df['speaker'] == speaker]
    successfully_downloaded = 0

    for _, row in tqdm(df_by_speaker.iterrows(), total=df_by_speaker.shape[0]):
    
        i, name, link = row
        if 'youtube' in link:
            try:
                output_path = os.path.join(BASE_PATH, row['speaker'], 'videos', row['video_fn'])
                if not (os.path.exists(os.path.dirname(output_path))):
                    os.makedirs(os.path.dirname(output_path))
                command = 'youtube-dl -o {temp_path} -f mp4 {link}'.format(link=link, temp_path=temp_output_path)
                res1 = call(command, shell=True)
                cam = cv2.VideoCapture(temp_output_path)
                if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
                    shutil.move(temp_output_path, output_path)
                else:
                    res2 = call('ffmpeg -i {} -r 30000/1001 -strict -2 {} -y'.format(temp_output_path,
                                                                                     output_path),
                                shell=True)
                successfully_downloaded += 1
            except Exception as e:
                print(e)
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
    print('Successfully downloaded {} out of {} videos for {}.'.format(successfully_downloaded,
                                                                       len(df_by_speaker), speaker))
    # print('Successfully downloaded:')
    # my_cmd = 'ls ' + os.path.join(BASE_PATH, row['speaker'], 'videos') + ' | wc -l'
    # os.system(my_cmd)
