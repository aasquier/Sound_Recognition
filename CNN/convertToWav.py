import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import scipy as sc
import numpy as np
from scipy.fftpack import fft, fftfreq
import librosa
import glob
import os

np.set_printoptions(threshold=np.nan) #Configure numpy's array printing to show everything

#Sample sound clips
CLIP = "1-137-A-32.wav"
CLIP2 = "1-9887-A-49.wav"

def main():
    data, sampleRate      = librosa.load(CLIP)
    mfcc, chroma, tonnetz = extractAudioFeatures(data, sampleRate)
    mel, contrast         = extractImageFeatures(data, sampleRate)

    #Audio features
    print "mfcc.shape: ", mfcc.shape
    print "chroma.shape: ", chroma.shape
    print "tonnetz: ", tonnetz.shape

    #Image features
    print "mel.shape: ", mel.shape
    print "contrast: ", contrast.shape



def extractAudioFeatures(data, sampleRate):
    stft    = np.abs(librosa.stft(data))
    mfccs   = np.mean(librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=40).T,axis=0)
    chroma  = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data),
    sr=sampleRate).T,axis=0)
    return mfccs,chroma,tonnetz


def extractImageFeatures(data, sampleRate):
    stft     = np.abs(librosa.stft(data))
    mel      = np.mean(librosa.feature.melspectrogram(data, sr=sampleRate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T,axis=0)

    return mel,contrast



def parseAudioFiles(parentDirectory,subDirectories,fileExt="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for subDirectory in subDirectories:
        for fileName in glob.glob(os.path.join(parentDirectory, subDirectory, fileExt)):
            try:
                data, sampleRate       = librosa.load(fileName)
                mel, contrast          = extractImageFeatures(data, sampleRate)
                mfccs, chroma, tonnetz = extractAudioFeatures(data, sampleRate)

            except Exception as e:
                print "Error encountered while parsing file: ", fn
                continue
            extFeatures = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features    = np.vstack([features,extFeatures])
            labels      = np.append(labels, fn.split('/')[2].split('-')[1])

    return np.array(features), np.array(labels, dtype = np.int)


''' Magically converts a .wav file into an array of integers '''
def convertWavToInt(fileName):
    soundData = wavfile.read(fileName)
    return soundData[0], soundData[1]


''' Converts a .wav file into a spectrogram'''
def convertWavToSpectrogram(fileName):
    sampleRate, samples = wavfile.read(fileName)
    frequencies, times, spectrogram = signal.spectrogram(samples, sampleRate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


main()
