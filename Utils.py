import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyaudio
import wave
import speech_recognition as s_r
import struct
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import os.path
from tensorflow.keras.models import load_model





def lower_sound(flag):
    """
    The function lowers the sound of the computer.

    Parameters:
        flag (bool): controls the option to lower and to return the sound.

    Returns:
        int: 1 if lowered and 0 if returned the sound.
     """
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.GetMute()
        volume.GetMasterVolumeLevel()
        volume.GetVolumeRange()
        if flag:
            volume.SetMasterVolumeLevel(-60.0, None)
            return 1
        else:
            volume.SetMasterVolumeLevel(-30.0, None)
            return 0


def is_silent(data, THRESHOLD):
    return max(data) < THRESHOLD


def record(count):
    """
    the function records until there`s no more sound.

    Parameters:
        count (bool): number of file.
    """
    # Get recording parameters
    BLOCKSIZE = 128
    RATE = 22050
    WIDTH = 2
    CHANNELS = 1
    LEN = 1 * RATE
    path= "temp/output"+str(count)+".wav"

    if not os.path.isdir('temp'):
        print('in')
        os.mkdir('temp')

    # Output wave file
    output_wf = wave.open(path, 'w')
    # output_wf = wave.open('222/-1_richard_22.wav', 'w')
    output_wf.setframerate(RATE)
    output_wf.setsampwidth(WIDTH)
    output_wf.setnchannels(CHANNELS)

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True)

    # Wait until voice detected
    while True:
        input_string = stream.read(BLOCKSIZE, exception_on_overflow=False)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)
        silent = is_silent(input_value, 1000)
        if not silent:
            break

        # Start recording
    print("Start")

    nBLOCK = int(LEN / BLOCKSIZE)
    numSilence = 0
    for n in range(0, nBLOCK + 1):

        if is_silent(input_value, 100):
            numSilence += 1
        #            output_value = np.zeros(BLOCKSIZE)

        output_value = np.array(input_value)

        if numSilence > 14:
            break

        output_value = output_value.astype(int)
        output_value = np.clip(output_value, -2 ** 15, 2 ** 15 - 1)

        ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
        output_wf.writeframes(ouput_string)

        input_string = stream.read(BLOCKSIZE, exception_on_overflow=False)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)

    print('Done')

    stream.stop_stream()
    stream.close()
    p.terminate()
    output_wf.close()




