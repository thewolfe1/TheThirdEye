import os
import pathlib

from gtts import gTTS
from pydub import AudioSegment
from pydub.utils import which
import subprocess


def save_file(string):
    """
    The function saves a mp3 file of the word that is given.

    Parameters:
        string (string): the word that will be generated.
    """
    pathlib.Path('GeneratedAudio/' + string).mkdir(parents=True, exist_ok=True)  # checks if dir exists
    i = 0
    while os.path.exists('GeneratedAudio/' + string + '/' + string + '-%s' % i + ".mp3"):
        i += 1
    tts.save('GeneratedAudio/' + string + '/' + string + '-%s' % i + ".mp3")


name=[]
with open('Files/Names.txt') as fp:
    for line in enumerate(fp):
        name.append(line)


for n in name:
    (int,string) = n

    tts = gTTS(text=string, lang='cs', slow=False)
    save_file('cs')

    tts = gTTS(text=string, lang='en', slow=False)
    save_file('en')

    tts = gTTS(text=string, lang='ar', slow=False)
    save_file('ar')

    tts = gTTS(text=string, lang='it', slow=False)
    save_file('it')

    tts = gTTS(text=string, lang='el', slow=False)
    save_file('el')

    tts = gTTS(text=string, lang='ca', slow=False)
    save_file('ca')

    tts = gTTS(text=string, lang='sv', slow=False)
    save_file('sv')

    tts = gTTS(text=string, lang='ru', slow=False)
    save_file('ru')

    tts = gTTS(text=string, lang='sk', slow=False)
    save_file('sk')

    tts = gTTS(text=string, lang='fr', slow=False)
    save_file('fr')

    tts = gTTS(text=string, lang='hi', slow=False)
    save_file('hi')

    tts = gTTS(text=string, lang='th', slow=False)
    save_file('th')

    tts = gTTS(text=string, lang='uk', slow=False)
    save_file('uk')

    tts = gTTS(text=string, lang='de', slow=False)
    save_file('de')

    tts = gTTS(text=string, lang='ko', slow=False)
    save_file('ko')

languages =['ar','ca','cs','de','el','en','fr','hi','it','ko','ru','sk','sv','th','uk']
owd = os.getcwd()
for n in languages:
    os.chdir(owd)
    path = "GeneratedAudio/" + n + '/'
    #Change working directory
    os.chdir(path)
    audio_files = os.listdir()
#    You dont need the number of files in the folder, just iterate over them directly using:
    for file in audio_files:
        #spliting the file into the name and the extension
        name, ext = os.path.splitext(file)
        if ext == ".mp3":
            #mp3_sound = AudioSegment.from_mp3(file)
            #rename them using the old name + ".wav"
            subprocess.call(['ffmpeg', '-i', file,
                             "{0}.wav".format(name)])
            #mp3_sound.export("{0}.wav".format(name), format="wav")
            os.remove(file)










