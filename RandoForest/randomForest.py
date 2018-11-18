import numpy as np
import extractFeatures as ef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

def main():
    accResults = np.zeros((10, 3))

    for i in range(10):
        audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets, videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets, audioTrainingExamplesNormalized, audioTestingExamplesNormalized, audioTrainingTargetsNormalized, audioTestingTargetsNormalized, videoTrainingExamplesNormalized, videoTestingExamplesNormalized, videoTrainingTargetsNormalized, videoTestingTargetsNormalized = ef.crossValidationIteration(i)

        fullTrainingExamples = np.concatenate((audioTrainingExamples, videoTrainingExamples))
        fullTestingExamples  = np.concatenate((audioTestingExamples, videoTestingExamples))
        fullTrainingTargets  = np.hstack((audioTrainingTargets, videoTrainingTargets))
        fullTestingTargets   = np.hstack((audioTestingTargets, videoTestingTargets))

        clfAudio = RandomForestClassifier(n_estimators=200)
        clfAudio.fit(audioTrainingExamples, audioTrainingTargets)
        audioTestPredictions = clfAudio.predict(audioTestingExamples)

        clfVideo = RandomForestClassifier(n_estimators=200)
        clfVideo.fit(videoTrainingExamples, videoTrainingTargets)
        videoTestPredictions = clfVideo.predict(videoTestingExamples)

        clfAll = RandomForestClassifier(n_estimators=200)
        clfAll.fit(fullTrainingExamples, fullTrainingTargets)
        fullTestPredictions = clfAll.predict(fullTestingExamples)

        audioTestAcc  = round((metrics.accuracy_score(audioTestingTargets, audioTestPredictions) * 100), 2)
        videoTestAcc  = round((metrics.accuracy_score(videoTestingTargets, videoTestPredictions) * 100), 2)
        fullTestAcc   = round((metrics.accuracy_score(fullTestingTargets, fullTestPredictions) * 100), 2)

        accResults[i, 0] = audioTestAcc
        accResults[i, 1] = videoTestAcc
        accResults[i, 2] = fullTestAcc

        print("\nAccuracy on audio test data for iteration " + str(i+1) + "    :   ", audioTestAcc, "%")
        print("Accuracy on video test data for iteration " + str(i+1) + "    :   ", videoTestAcc, "%")
        print("Accuracy on combined test data for iteration " + str(i+1) + " :   ", fullTestAcc, "%")

        # calculateConfusionMatrix(audioTestingTargets, audioTestPredictions)
        # calculateConfusionMatrix(videoTestingTargets, videoTestPredictions)
        # calculateConfusionMatrix(fullTestingTargets, fullTestPredictions)

    accTotals = np.sum(accResults, axis=0)
    accTotals /= 10.0

    print("\nAverage accuracy on audio test data:      ", round((accTotals[0]), 2), "%")
    print("Average accuracy on video test data:      ", round((accTotals[1]), 2), "%")
    print("Average accuracy on combined test data:   ", round((accTotals[2]), 2), "%")


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
