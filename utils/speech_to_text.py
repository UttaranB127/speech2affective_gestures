# importing libraries
import glob
import os
import shutil

import speech_recognition as sr

from pydub.silence import split_on_silence

from pydub import AudioSegment


def change_speed(audio, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    })
    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(audio.frame_rate)


# a function that splits the audio file into chunks
# and applies speech recognition
def silence_based_conversion(_audio_file, _text_file, chunk_dir, min_silence_len, silence_dbf_thresh, audio_speed=1.0):
    # open the audio file stored in
    # the local system as a wav file.
    audio = AudioSegment.from_wav(_audio_file)

    # split track where silence is 0.5 seconds
    # or more and get chunks
    chunks = split_on_silence(audio,
                              # must be silent for at least 0.5 seconds
                              # or 500 ms. adjust this value based on user
                              # requirement. if the speaker stays silent for
                              # longer, increase this value. else, decrease it.
                              min_silence_len=min_silence_len,

                              # consider it silent if quieter than -16 dBFS
                              # adjust this per requirement
                              silence_thresh=silence_dbf_thresh
                              )

    # process each chunk
    num_chunks = len(chunks)
    for chunk_idx, chunk in enumerate(chunks):

        # Create silence chunk
        chunk_silent = AudioSegment.silent(duration=min_silence_len)

        # add silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        audio_chunk = change_speed(audio_chunk, speed=audio_speed)

        # the name of the chunk file
        chunk_file = os.path.join(chunk_dir, 'chunk_{:03d}'.format(chunk_idx) + '.wav')

        # export audio chunk and save it in
        # the current directory.
        # specify the bitrate to be 192 k
        audio_chunk.export(chunk_file, bitrate='192k', format='wav')

        # create a speech recognition object
        r = sr.Recognizer()

        # recognize the chunk
        with sr.AudioFile(chunk_file) as source:
            # remove this if it is not working
            # correctly.
            r.adjust_for_ambient_noise(source)
            audio_listened = r.listen(source)

        print('\ttext from chunk {:>6}/{:>6}:\t'.format(chunk_idx, num_chunks - 1), end='')
        try:
            # try converting it to text
            rec = r.recognize_google(audio_listened)

            # open a file where we will concatenate
            # and store the recognized text
            with open(_text_file, 'w+') as tf:
                tf.write(rec + ' ')
            print('{}'.format(rec))

        # catch any errors.
        except sr.UnknownValueError:
            print('Error! Could not understand audio')

        except sr.RequestError:
            print('Error! Could not request results. check your internet connection')


speakers = ['almaram', 'angelica', 'chemistry', 'conan', 'ellen', 'jon', 'oliver', 'rock', 'seth', 'shelly']
base_dir = '/media/uttaran/repo1/s2g_ginosar/data'
tmp_dir = '/media/uttaran/repo1/s2g_ginosar/data/tmp'
os.makedirs(tmp_dir, exist_ok=True)
for speaker in speakers:
    audio_dir = os.path.join(base_dir, speaker, 'train/audio')
    text_dir = os.path.join(base_dir, speaker, 'train/text')
    os.makedirs(text_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    num_audio_files = len(audio_files)
    for file_idx, audio_file in enumerate(audio_files):
        file_base = '.wav'.join(audio_file.split('/')[-1].split('.wav')[:-1])
        print('Speaker: {}. Audio_file {:>6}/{:>6}: {}.'.format(speaker, file_idx + 1, num_audio_files, file_base))
        text_file = os.path.join(text_dir, file_base + '.txt')
        if not os.path.exists(text_file):
            silence_based_conversion(audio_file, text_file, chunk_dir=tmp_dir,
                                     min_silence_len=1000, silence_dbf_thresh=-32, audio_speed=0.84)
        print()
shutil.rmtree(tmp_dir)
