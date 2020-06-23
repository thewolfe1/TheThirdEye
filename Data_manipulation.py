import librosa
import numpy as np
from scipy.io import wavfile
import cv2
import pathlib
import os


def set_path(folder):
    """
    The function removes the start of the audio file.

    Parameters:
        folder (string): the folder to save the file at.
    Returns:
        String: the file to read the names from.
    """
    if folder == 'speaker':
        return'Files/People_Names.txt'
    return 'Files/Names.txt'


#10,20,30,40,50,60,70
def rotate(file,num,count,folder):
    """
    The function rotates the audio file.

    Parameters:
        file (string): file path.
        num (int): rotation level.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """

    y, sr = librosa.load(file)  # Read audio
    y = np.roll(y, sr * num)
    path=set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/rotate').mkdir(parents=True,exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/rotate/{}{}.wav".format(line[1].rstrip(),count,num), sr, y)  # Write audio
                count+=1

#1.1,1.2,1.3,1.4,1.5
def faster(file,num,count,folder):
    """
    The function speeds up the audio file.

    Parameters:
        file (string): file path.
        num (int): speed.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)
    y_fast = librosa.effects.time_stretch(y, num)
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/faster').mkdir(parents=True,exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/faster/{}{}.wav".format(line[1].rstrip(),count,num), sr, y_fast)  # Write audio
                count+=1


#0.5,1
def slower(file,num,count,folder):
    """
    The function slows down the audio file.

    Parameters:
        file (string): file path.
        num (int): speed.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)
    y_slow = librosa.effects.time_stretch(y, 0.5)
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder + '/' + line[1].rstrip() + '/slower').mkdir(parents=True,
                                                                                exist_ok=True)  # checks if dir exists
                wavfile.write(folder + "/{}/slower/{}{}.wav".format(line[1].rstrip(), count, num), sr, y_slow)  # Write audio
                count += 1

#1.2,1.4,1.6
def resize(file,num,count,folder):
    """
    The function resizes the vector of the audio file.

    Parameters:
        file (string): file path.
        num (int): size.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)  # Read audio
    ly = len(y)
    y_tune = cv2.resize(y, (1, int(len(y) * num))).squeeze()
    lc = len(y_tune) - ly
    y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/resize').mkdir(parents=True,exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/resize/{}{}.wav".format(line[1].rstrip(),count,num), sr, y) #Write audio


#0.001,0.006,0.011,0.016
def addNoise(file, noise_factor,count,folder):
    """
    The function adds noise to the audio file.

    Parameters:
        file (string): file path.
        num (int): noise value.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(y[0]))
    path=set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/noise').mkdir(parents=True, exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/noise//{}{}.wav".format(line[1].rstrip(),count,noise_factor),sr,augmented_data)

#2000
def removeStart(file,num,count,folder):
    """
    The function removes the start of the audio file.

    Parameters:
        file (string): file path.
        num (int): time.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)  # Read audio
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/remove').mkdir(parents=True,exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/remove/start{}.wav".format(line[1].rstrip(),count), sr, y[num:sr])  # Write audio

#-11000
def removeEnd(file,num,count,folder):
    """
    The function removes the end of the audio file.

    Parameters:
        file (string): file path.
        num (int): time.
        count (int): number of file.
        folder (string): the folder to save the file at.
    """
    y, sr = librosa.load(file)  # Read audio
    path = set_path(folder)
    with open(path) as fp:
        for line in enumerate(fp):
            if line[1].rstrip() in file:
                pathlib.Path(folder+'/' + line[1].rstrip() + '/remove').mkdir(parents=True,exist_ok=True)  # checks if dir exists
                wavfile.write(folder+"/{}/remove/end{}.wav".format(line[1].rstrip(),count), sr, y[0:sr-num])  # Write audio


