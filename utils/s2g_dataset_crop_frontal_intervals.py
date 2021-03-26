import argparse
from tqdm import tqdm
import subprocess
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', default='/media/uttaran/repo1/data/s2g',
                    help='base folder path of dataset')
parser.add_argument('-output_path', '--output_path',
                    default='output directory to save cropped intervals')
parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

args = parser.parse_args()


def save_interval(_in_file, _start, _end, _out_file):
    cmd = 'ffmpeg -i {} -ss {} -to {} -strict -2 {} -y'.format(_in_file, _start, _end, _out_file)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    df_intervals = pd.read_csv(os.path.join(args.base_path, 'intervals_df.csv'))
    if args.speaker:
        df_intervals = df_intervals[df_intervals['speaker'] == args.speaker]

    for _, interval in tqdm(df_intervals.iterrows(), total=len(df_intervals)):
        try:
            start_time = str(pd.to_datetime(interval['start_time']).time())
            end_time = str(pd.to_datetime(interval['end_time']).time())
            in_file = os.path.join(args.base_path, interval['speaker'], 'videos', interval['video_fn'])
            out_file = os.path.join(args.output_path,
                                    '{}_{}_{}-{}.mp4'.format(interval['speaker'], interval['video_fn'],
                                                             str(start_time), str(end_time)))
            print(in_file, out_file)
            save_interval(in_file, str(start_time), str(end_time), out_file)
        except Exception as e:
            print(e)
            print('could not crop interval: {}'.format(interval))
