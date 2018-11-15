import re
import os
import sys
import glob
import errno
import librosa
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

np.set_printoptions(threshold=np.nan) #Configure numpy's array printing to show everything

DATA_PATH      = "data/audio/*.wav"
N_MFCC         = 117
AUDIO_FEATURES = 135
IMAGE_FEATURES = 135
READ_ALL = -1


def main():
    audioFeatures, imageFeatures = parseAudioFiles(DATA_PATH, READ_ALL)
    saveAsCSV(audioFeatures, imageFeatures)

'''
Reads and parses a specified amount of .wav files from a directory,
strips the features from each file, and stores them into a .csv file.
Returns
'''
def parseAudioFiles(path, amountToRead):
    files = glob.glob(path)
    filesAmt = len(files)
    audioFeatures = np.zeros((filesAmt, AUDIO_FEATURES + 1), float)  #Add one for the label
    imageFeatures = np.zeros((filesAmt, IMAGE_FEATURES + 1), float)
    i = 0
    for name in files:
        if (".wav" in name) and ((i < amountToRead) or (amountToRead == -1)):
            if i % 10 == 0:
                print("Files read: ", i)
            data, sampleRate = librosa.load(name)
            mfcc, chroma, tonnetz = extractAudioFeatures(data, sampleRate)
            mel, contrast = extractImageFeatures(data, sampleRate)
            label = stripLabel(name)
            audioFeatures[i] = np.hstack((label, mfcc, chroma, tonnetz))
            imageFeatures[i] = np.hstack((label, mel, contrast))
            i += 1
        else:
            break
    return audioFeatures, imageFeatures


''' Takes two arrays of features and saves them into .csv files '''
def saveAsCSV(audioFeatures, imageFeatures):
    np.savetxt("audioData.csv", audioFeatures, delimiter=",")
    np.savetxt("imageData.csv", imageFeatures, delimiter=",")


''' Returns the label by stripping it from the filename '''
def stripLabel(fileName):
    regX  = re.compile('(data/audio/\d+-\d+-([A-Z])-)(\d+)(\.wav)')
    match = regX.match(fileName)
    return match.group(3)


''' Returns three arrays that make up the audio features '''
def extractAudioFeatures(data, sampleRate):
    stft    = np.abs(librosa.stft(data))
    mfccs   = np.mean(librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=N_MFCC).T,axis=0)
    chroma  = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data),
    sr=sampleRate).T,axis=0)

    return mfccs,chroma,tonnetz


''' Returns two arrays that make up the audio features '''
def extractImageFeatures(data, sampleRate):
    stft     = np.abs(librosa.stft(data))
    mel      = np.mean(librosa.feature.melspectrogram(data, sr=sampleRate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T,axis=0)

    return mel,contrast


''' Converts a .wav file into a spectrogram '''
def convertWavToSpectrogram(fileName):
    sampleRate, samples = wavfile.read(fileName)
    frequencies, times, spectrogram = signal.spectrogram(samples, sampleRate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


main()
