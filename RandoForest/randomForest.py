import numpy as np
import extractFeatures as ef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb



def main():
    rawDataAudio, rawDataAudioNormalized, rawDataVideo, rawDataVideoNormalized = ef.readDataFiles()

    for i in range(10):
        audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets, videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets = ef.crossValidationIteration(rawDataAudio, rawDataVideo, i)

        fullTrainingExamples = np.concatenate((audioTrainingExamples, videoTrainingExamples))
        fullTestingExamples  = np.concatenate((audioTestingExamples, videoTestingExamples))
        fullTrainingTargets  = np.hstack((audioTrainingTargets, videoTrainingTargets))
        fullTestingTargets   = np.hstack((audioTestingTargets, videoTestingTargets))


        clfAudio = RandomForestClassifier(n_estimators=1000)
        clfAudio.fit(audioTrainingExamples, audioTrainingTargets)
        audioTestPredictions = clfAudio.predict(audioTestingExamples)
        audioTrainPredictions = clfAudio.predict(audioTrainingExamples)


        clfVideo = RandomForestClassifier(n_estimators=1000)
        clfVideo.fit(videoTrainingExamples, videoTrainingTargets)
        videoTestPredictions = clfVideo.predict(videoTestingExamples)
        videoTrainPredictions = clfVideo.predict(videoTrainingExamples)

        clfAll = RandomForestClassifier(n_estimators=1000)
        clfAll.fit(fullTrainingExamples, fullTrainingTargets)
        fullTestPredictions = clfAll.predict(fullTestingExamples)
        fullTrainPredictions = clfAll.predict(fullTrainingExamples)

        print("Accuracy on audio test data:        ", (round((metrics.accuracy_score(audioTestingTargets, audioTestPredictions) * 100), 4)), "%")
        print("Accuracy on audio training data:    ", (round((metrics.accuracy_score(audioTrainingTargets, audioTrainPredictions) * 100), 4)), "%")

        print("Accuracy on video test data:        ", (round((metrics.accuracy_score(videoTestingTargets, videoTestPredictions) * 100), 4)), "%")
        print("Accuracy on video training data:    ", (round((metrics.accuracy_score(videoTrainingTargets, videoTrainPredictions) * 100), 4)), "%")

        print("Accuracy on combined test data:     ", (round((metrics.accuracy_score(fullTestingTargets, fullTestPredictions) * 100), 4)), "%")
        print("Accuracy on combined training data: ", (round((metrics.accuracy_score(fullTrainingTargets, fullTrainPredictions) * 100), 4)), "%")

        calculateConfusionMatrix(audioTestingTargets, audioTestPredictions)
        calculateConfusionMatrix(videoTestingTargets, videoTestPredictions)
        calculateConfusionMatrix(fullTestingTargets, fullTestPredictions)

    return


# use panda, pyplot, and seaborn to plot a confusion matrix
def calculateConfusionMatrix(targets, predictions):
    confusion = np.zeros((50,50), int)
    # populate the confusion matrix by adding 1 to the elements corresponding to (t, p)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    # create a panda data frame and feed it to the seaborn heatmap function
    confusionMatrix = pd.DataFrame(confusion)
    plt.figure(figsize = (10,7))
    sb.heatmap(confusionMatrix, annot=True, fmt='g')
    plt.show()

    return


main()