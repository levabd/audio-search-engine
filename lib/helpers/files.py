import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop


def loadfile(filename):
    # pydub does not support 24-bit wav files, use wavio when this occurs
    try:
        audiofile = AudioSegment.from_file(filename)

        data = np.fromstring(audiofile._data, np.int16)

        channels = []
        for chn in range(audiofile.channels):
            channels.append(data[chn::audiofile.channels])

        fs = audiofile.frame_rate
    except audioop.error:
        fs, _, audiofile = wavio.readwav(filename)

        audiofile = audiofile.T
        audiofile = audiofile.astype(np.int16)

        channels = []
        for chn in audiofile:
            channels.append(chn)
    
    return channels, fs