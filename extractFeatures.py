import re
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing


'''
There are three main functions you can do with this program:
1. Extract and save the data to a .csv file
2. Extract, normalize, and save the data to a .csv file
3. Read in from the .csv file with readDataFile() (import functino into project)

=> To import from another directory:
import sys
sys.path.insert(0, 'path/to/directory/with/extractFeatures')
from extractFeatures import readDataFile
'''

N_MFCC          = 117
AUDIO_FEATURES  = 135
IMAGE_FEATURES  = 135
READ_ALL        = -1
TRAINING_M      = 6000     # How many examples do we want here to divide the training and test sets??


''' Extracts features and saves into a .csv file '''
def extractFeatures():
    filePath = "data/audio/fold1/*.wav"
    audioFeatures, imageFeatures = parseAudioFiles(filePath, READ_ALL)
    csvAudioFile = "./data/audioDataFold1.csv"
    csvVideoFile = "./data/imageDataFold1.csv"
    saveAsCSV(audioFeatures, imageFeatures, csvAudioFile, csvVideoFile)


''' Extracts and normalizes features and saves into a .csv file '''
def extractAndNormalizeFeatures():
    filePath = "data/audio/fold1/*.wav"
    audioFeatures, imageFeatures = parseAudioFiles(filePath, READ_ALL)
    audioFeatures, imageFeatures = normalizeData(audioFeatures, imageFeatures)
    csvAudioFile = "./data/audioDataFold1Normalized.csv"
    csvVideoFile = "./data/imageDataFold1Normalized.csv"
    saveAsCSV(audioFeatures, imageFeatures, csvAudioFile, csvVideoFile)


'''
Reads and parses a specified amount of .wav files from a directory,
strips the features from each file, and stores them into a .csv file.
Returns
'''
def parseAudioFiles(path, amountToRead):
    files         = glob.glob(path)
    filesAmt      = len(files)
    audioFeatures = np.zeros((filesAmt, AUDIO_FEATURES + 1), float)  # Add one for the label
    imageFeatures = np.zeros((filesAmt, IMAGE_FEATURES + 1), float)
    i = 0
    for name in files:
        if (".wav" in name) and ((i < amountToRead) or (amountToRead == -1)):
            # if i % 10 == 0:
            print("Files read: ", i)
            if (i == 496) or (i == 565) or (i == 711):
                i += 1
                continue
            data, sampleRate      = librosa.load(name)
            mfcc, chroma, tonnetz = extractAudioFeatures(data, sampleRate)
            mel, contrast         = extractImageFeatures(data, sampleRate)
            label                 = stripLabel(name)
            audioFeatures[i]      = np.hstack((label, mfcc, chroma, tonnetz))
            imageFeatures[i]      = np.hstack((label, mel, contrast))
            i += 1
        else:
            break

    return audioFeatures, imageFeatures


''' Takes two arrays of features and saves them into .csv files '''
def saveAsCSV(audioFeatures, imageFeatures, audioFileName, imageFileName):
    np.savetxt(audioFileName, audioFeatures, delimiter=",")
    np.savetxt(imageFileName, imageFeatures, delimiter=",")


''' Returns the label by stripping it from the filename '''
def stripLabel(fileName):
    regX  = re.compile('(data/audio/fold\d+/\d+-)(\d)(-\d+-\d+.wav)')
    match = regX.match(fileName)
    print(match.group(2))
    return match.group(2)


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


''' Returns normalized versions of the each array '''
def normalizeData(audioData, imageData):
    examples         = len(audioData)
    audioDataWOLabel = np.zeros((examples, AUDIO_FEATURES), float)
    imageDataWOLabel = np.zeros((examples, IMAGE_FEATURES), float)
    audioNormalized  = np.zeros((examples, AUDIO_FEATURES + 1), float)
    imageNormalized  = np.zeros((examples, IMAGE_FEATURES + 1), float)

    for i in range(len(audioData)):
        audioDataWOLabel[i] = audioData[i][1:]
        imageDataWOLabel[i] = imageData[i][1:]

    audioDataWOLabel = preprocessing.normalize(audioDataWOLabel)
    imageDataWOLabel = preprocessing.normalize(imageDataWOLabel)

    for i in range(len(audioData)):
        audioNormalized[i] = np.hstack((audioData[i][:1], audioDataWOLabel[i]))
        imageNormalized[i] = np.hstack((imageData[i][:1], imageDataWOLabel[i]))

    return audioNormalized, imageNormalized
'''
Reads a file of examples, splits it into two sets (one for
training and one for testing), and then peels off the targets
(last value in each row)
Returns the training set, testing set, training targets, testing targets
'''
def readDataFile(fileName):
    rawData = np.genfromtxt(fileName, delimiter=',')
    np.random.shuffle(rawData)

    #Format the training examples
    trainingExamples = rawData[:TRAINING_M]
    trainingExamples, trainingTargets = stripTargets(trainingExamples)

    #Format the testing examples
    testingExamples = rawData[TRAINING_M:]
    testingExamples, testingTargets = stripTargets(testingExamples)

    return trainingExamples, trainingTargets, testingExamples, testingTargets

def readDataFiles():
    rawDataAudio           = np.empty(10, float)
    rawDataAudioNormalized = np.empty(10, float)
    rawDataVideo           = np.empty(10, float)
    rawDataVideoNormalized = np.empty(10, float)

    for i in range(10):
        audioFileName             = "./data/audio/audioDataFold" + str(i+1) + ".csv"
        audioFileNameNormal       = "./data/audio/audioDataFold" + str(i+1) + "Normalized.csv"
        videoFileName             = "./data/audio/imageDataFold" + str(i+1) + ".csv"
        videoFileNameNormal       = "./data/audio/imageDataFold" + str(i+1) + "Normalized.csv"
        rawDataAudio[i]           = np.genfromtxt(audioFileName, delimiter=',')
        rawDataAudioNormalized[i] = np.genfromtxt(audioFileNameNormal, delimiter=',')
        rawDataVideo[i]           = np.genfromtxt(videoFileName, delimiter=',')
        rawDataVideoNormalized[i] = np.genfromtxt(videoFileNameNormal, delimiter=',')

    return rawDataAudio, rawDataAudioNormalized, rawDataVideo, rawDataVideoNormalized


def crossValidationIteration(rawDataAudio, rawDataVideo, iteration):
    audioTestingExamples  = rawDataAudio[iteration]
    audioTrainingExamples = np.delete(rawDataAudio, iteration, axis=0)
    videoTestingExamples  = rawDataVideo[iteration]
    videoTrainingExamples = np.delete(rawDataVideo, iteration, axis=0)

    np.random.shuffle(audioTestingExamples)
    np.random.shuffle(audioTrainingExamples)
    np.random.shuffle(videoTestingExamples)
    np.random.shuffle(videoTrainingExamples)

    audioTestingTargets  = stripTargets(audioTestingExamples)
    audioTrainingTargets = stripTargets(audioTrainingExamples)
    videoTestingTargets  = stripTargets(videoTestingExamples)
    videoTrainingTargets = stripTargets(videoTrainingExamples)

    return  audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets, videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets

'''
Removes the target from each row (the last index) and
and returns them in an array.
@param examples: Rows of features where the last element is the target
'''
def stripTargets(examplesRaw):
    examplesAmt     = len(examplesRaw)                      # Number of examples
    featuresAmt     = len(examplesRaw[0]) - 1               # Amount of features - target
    examples        = np.zeros((examplesAmt, featuresAmt))
    targets         = np.zeros(examplesAmt, int)            # Targets for each examples

    for i in range(examplesAmt):
        targets[i]  = int(examplesRaw[i, 0])                # Strip the target from the row
        examples[i] = (examplesRaw[i])[1:]                  # Remove the last element

    return examples, targets


''' Converts a .wav file into a spectrogram '''
def convertWavToSpectrogram(fileName):
    sampleRate, samples             = wavfile.read(fileName)
    frequencies, times, spectrogram = signal.spectrogram(samples, sampleRate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def main():
    extractFeatures()
    extractAndNormalizeFeatures()

    return


main()