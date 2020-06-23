from __future__ import print_function

import pathlib

import librosa
import numpy as np
import glob
from random import randint

import glob

import librosa
import numpy as np

import librosa.display

import soundfile as sf
from presets import Preset
from python_speech_features import mfcc,logfbank
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import matplotlib.pyplot as plt

# Import the Preset class
from presets import Preset
import librosa as _librosa
import librosa.display as _display

from Data_manipulation import rotate,addNoise,removeEnd,removeStart,resize,set_path,slower,faster



def audio2vector(file_path, max_pad_len=400):
    """
    The function extracts the vector of the audio file using librosas mfcc function.

    Parameters:
        file_path (string): the audio file path.
        max_pad_len (int): padding length.
    Returns:
        numpy array: the vector of the audio file.
    """
    # read the audio file
    audio, sr = librosa.load(file_path, mono=True)

    audio = removeSilence(audio)

    # reduce the shape
    audio = audio[::3]

    # extract the audio embeddings using MFCC
    mfccs = librosa.feature.mfcc(audio, sr=sr,n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfccsscaled = np.mean(mfccs.T, axis=0)
    # as the audio embeddings length varies for different audio, we keep the maximum length as 400
    # pad them with zeros
    #pad_width = max_pad_len - mfccsscaled.shape[1]
    #mfcc = np.pad(mfccsscaled, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccsscaled

def audio2vector2(file_path, max_pad_len=400):
    """
    The function extracts the vector of the audio file using librosas melspectrogram function.

    Parameters:
        file_path (string): the audio file path.
        max_pad_len (int): padding length.
    Returns:
        numpy array: the vector of the audio file.
    """
    y, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    y_trimmed, _ = librosa.effects.trim(y)
    y_with_silence = np.copy(y)
    y_with_silence[10000:30000] = 0
    y_with_silence[40000:60000] = 0
    # it returns an array of [start, end] elements (which are non-silent)
    non_silent_intervals = librosa.effects.split(y_with_silence)
    # Remix could be useful - it re-orders time intervals removing silence
    y_remixed = librosa.effects.remix(y_with_silence, non_silent_intervals)
    S = librosa.feature.melspectrogram(y=y, n_mels=128, fmax=8000)
    pad_width = max_pad_len - S.shape[1]
    padded = np.pad(S, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return padded



def audio2Image(name,path,num):
    """
    The function converts the spectogram of the audio file into an image.

    Parameters:
        name (string): the name of the file.
        path (string): the path of the file.
        num (int): number of the image.
    """
    pathlib.Path(f'data/test/{name}').mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(path, mono=True, duration=5)
    print(y.shape)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB')
    plt.axis('off')
    plt.savefig(f'data/test/{name}/{name.replace(".", "")}{num}.png')
    plt.clf()



def removeSilence(x):
    """
    The function removes silence from the audio file.

    Parameters:
        x (string): vector.

    Returns:
        numpy array: the vector without silence.
    """
    y = librosa.effects.split(x, top_db=30)
    l = []
    for i in y:
        l.append(x[i[0]:i[1]])
    return np.concatenate(l, axis=0)

# set training data
def get_training_data():
    """
    The function reads all the audio files and extracts their features.

    Returns:
        tuple: numpy array of vector and numpy array of labels.
    """
    pairs = []
    labels = []
    original = []
    # read labels file and get all files
    with open('Files/People_Names.txt') as fp:
        for line in enumerate(fp):
            original.append(glob.glob('recordings/{}/*.wav'.format(line[1].strip())))
            original.append(glob.glob('recordings/expend/noise/{}/*.wav'.format(line[1].strip())))
            original.append(glob.glob('recordings/expend/remove/{}/*.wav'.format(line[1].strip())))
            original.append(glob.glob('recordings/expend/resize/{}/*.wav'.format(line[1].strip())))
            original.append(glob.glob('recordings/expend/rotate/{}/*.wav'.format(line[1].strip())))

    # make list of lists to one flat list
    original = [item for sublist in original for item in sublist]
    print(len(original))
    # original = glob.glob('test/jackson/*.wav')
    # good = glob.glob('test/sub/*.wav')
    good = original
    bad = glob.glob('recordings/other/*.wav')

    # np.random.shuffle(good)
    # np.random.shuffle(bad)
    # if we want equal size of good and bad change to min
    for i in range(min(len(bad), len(good))):
        # imposite pair
        if (i % 2) == 0:
            pairs.append([audio2vector(original[randint(0, len(original) - 1)]), audio2vector(bad[i])])
            labels.append(0)

        # genuine pair
        else:
            pairs.append([audio2vector(original[randint(0, len(original) - 1)]), audio2vector(good[i])])
            labels.append(1)

    return np.array(pairs), np.array(labels)


#get_training_data()

def rotateAudio(folder):
    """
    The function rotates the audio files.

    Parameters:
        folder (string): the folder to save the files at.
    """
    ranges=[10,20,30,40,50,60,70]
    recordings=[]
    count=0
    path = set_path(folder)
    if folder is 'speech':
        with open(path) as fp:
            for line in enumerate(fp):
                recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    for i in recordings:
        for j in i:
            for k in ranges:
                rotate(j,k,count,folder)
            count+=1
        count=0

def fastAudio(folder):
    """
    The function speeds up the audio files.

    Parameters:
        folder (string): the folder to save the files at.
    """
    ranges=[1.1,1.2,1.3,1.4,1.5]
    recordings=[]
    count=0
    path = set_path(folder)
    if folder is 'speech':
        with open(path) as fp:
            for line in enumerate(fp):
                recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    for i in recordings:
        for j in i:
            for k in ranges:
                faster(j,k,count,folder)
            count+=1
        count=0

def slowAudio(folder):
    """
    The function slows down the audio files.

    Parameters:
        folder (string): the folder to save the files at.
    """
    ranges=[0.5,1]
    recordings=[]
    count=0
    path = set_path(folder)
    if folder is 'speech':
        with open(path) as fp:
            for line in enumerate(fp):
                recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    for i in recordings:
        for j in i:
            for k in ranges:
                slower(j,k,count,folder)
            count+=1
        count=0

def resizeAudio(folder):
    """
    The function resizes the audio files vectors.

    Parameters:
        folder (string): the folder to save the files at.
    """
    ranges=[1.2,1.4,1.6]
    recordings=[]
    count=0
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    for i in recordings:
        for j in i:
            for k in ranges:
                resize(j,k,count,folder)
            count+=1
        count=0

def addNoiseAudio(folder):
    """
    The function adds noise to the audio files.

    Parameters:
        folder (string): the folder to save the files at.
    """
    ranges=[0.001,0.006,0.011,0.016]
    recordings=[]
    count=0
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    print(recordings)
    for i in recordings:
        for j in i:
            for k in ranges:
                addNoise(j,k,count,folder)
            count+=1
        count=0

def removePartAudio(folder):
    """
    The function removes the start and the end of the audio files.

    Parameters:
        folder (string): the folder to save the files at.
    """
    start=2000
    end=-11000
    recordings=[]
    count=0
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            recordings.append(glob.glob(folder+'/{}/*.wav'.format(line[1].strip())))
    print(recordings)
    for i in recordings:
        for j in i:
            removeStart(j,start,count,folder)
            removeEnd(j,end,count,folder)
            count+=1
        count=0



if __name__== "__main__":
    rotateAudio('speech')
    addNoiseAudio('speech')
    resizeAudio('speech')
    removePartAudio('speech')
    slowAudio('speech')
    fastAudio('speech')

