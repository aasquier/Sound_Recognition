import numpy as np
import extractFeatures as ef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


def main():
    pd.set_option('display.max_rows', 1000)
    accResults = np.zeros((10, 3))
    # accResults = np.zeros((10, 6))
    audioFeature_imp = np.zeros(10)
    videoFeature_imp = np.zeros(10)
    fullFeature_imp  = np.zeros(10)

    for i in range(10):
        audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets, videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets, audioTrainingExamplesNormalized, audioTestingExamplesNormalized, audioTrainingTargetsNormalized, audioTestingTargetsNormalized, videoTrainingExamplesNormalized, videoTestingExamplesNormalized, videoTrainingTargetsNormalized, videoTestingTargetsNormalized = ef.crossValidationIteration(i)

        fullTrainingExamples = np.concatenate((audioTrainingExamples, videoTrainingExamples))
        fullTestingExamples  = np.concatenate((audioTestingExamples, videoTestingExamples))
        # fullTrainingExamplesNormalized = np.concatenate((audioTrainingExamplesNormalized, videoTrainingExamplesNormalized))
        # fullTestingExamplesNormalized  = np.concatenate((audioTestingExamplesNormalized, videoTestingExamplesNormalized))
        fullTrainingTargets  = np.hstack((audioTrainingTargets, videoTrainingTargets))
        fullTestingTargets   = np.hstack((audioTestingTargets, videoTestingTargets))
        # fullTrainingTargetsNormalized  = np.hstack((audioTrainingTargetsNormalized, videoTrainingTargetsNormalized))
        # fullTestingTargetsNormalized   = np.hstack((audioTestingTargetsNormalized, videoTestingTargetsNormalized))

        clfAudio = RandomForestClassifier(n_estimators=100)
        clfAudio.fit(audioTrainingExamples, audioTrainingTargets)
        audioTestPredictions = clfAudio.predict(audioTestingExamples)
        audioFeature_imp = pd.Series(clfAudio.feature_importances_).sort_values(ascending=False)
        # audioFeatures = np.array(audioFeature_imp)
        relevantAudioFeatures = np.empty(0)
        for j in range(len(audioFeature_imp)):
            if audioFeature_imp[j] >= 0.0065:
                relevantAudioFeatures = np.hstack((relevantAudioFeatures, j))
        print(relevantAudioFeatures)
        # print(relevantAudioFeatures)

        sb.barplot(x=audioFeature_imp.index, y=audioFeature_imp)
        plt.ylabel('Feature Importance Score')
        plt.xlabel('Audio Features')
        plt.title("Visualizing Important Features")
        plt.show()

        clfVideo = RandomForestClassifier(n_estimators=100)
        clfVideo.fit(videoTrainingExamples, videoTrainingTargets)
        videoTestPredictions = clfVideo.predict(videoTestingExamples)
        videoFeature_imp = pd.Series(clfVideo.feature_importances_).sort_values(ascending=True)
        videoFeatures = np.array(videoFeature_imp)
        relevantVideoFeatures = np.empty(0)
        for j in range(len(videoFeature_imp)):
            if videoFeature_imp[j] >= 0.0065:
                relevantVideoFeatures = np.hstack((relevantVideoFeatures, j))
        print(relevantVideoFeatures)
        # print(relevantVideoFeatures)

        sb.barplot(x=videoFeature_imp.index, y=videoFeature_imp)
        plt.ylabel('Feature Importance Score')
        plt.xlabel('Video Features')
        plt.title("Visualizing Important Features")
        plt.show()

        clfAll = RandomForestClassifier(n_estimators=100)
        clfAll.fit(fullTrainingExamples, fullTrainingTargets)
        fullTestPredictions = clfAll.predict(fullTestingExamples)
        fullFeature_imp = pd.Series(clfAll.feature_importances_).sort_values(ascending=True)
        fullFeatures = np.array(fullFeature_imp)
        relevantFullFeatures = np.empty(0)
        for j in range(len(fullFeature_imp)):
            if fullFeature_imp[j] >= 0.0065:
                relevantFullFeatures = np.hstack((relevantFullFeatures, j))
        print(relevantFullFeatures)
        # print(relevantFullFeatures)

        sb.barplot(x=fullFeature_imp.index, y=fullFeature_imp)
        plt.ylabel('Feature Importance Score')
        plt.xlabel('Combined Features')
        plt.title("Visualizing Important Features")
        plt.show()

        # clfAudioNormalized = RandomForestClassifier(n_estimators=30)
        # clfAudioNormalized.fit(audioTrainingExamplesNormalized, audioTrainingTargetsNormalized)
        # audioTestPredictionsNormalized = clfAudio.predict(audioTestingExamplesNormalized)
        #
        # clfVideoNormalized = RandomForestClassifier(n_estimators=30)
        # clfVideoNormalized.fit(videoTrainingExamplesNormalized, videoTrainingTargetsNormalized)
        # videoTestPredictionsNormalized = clfVideo.predict(videoTestingExamplesNormalized)
        #
        # clfAllNormalized = RandomForestClassifier(n_estimators=30)
        # clfAllNormalized.fit(fullTrainingExamplesNormalized, fullTrainingTargetsNormalized)
        # fullTestPredictionsNormalized = clfAll.predict(fullTestingExamplesNormalized)

        audioTestAcc      = round((metrics.accuracy_score(audioTestingTargets, audioTestPredictions) * 100), 2)
        videoTestAcc      = round((metrics.accuracy_score(videoTestingTargets, videoTestPredictions) * 100), 2)
        fullTestAcc       = round((metrics.accuracy_score(fullTestingTargets, fullTestPredictions) * 100), 2)
        # audioTestAccNorm  = round((metrics.accuracy_score(audioTestingTargetsNormalized, audioTestPredictionsNormalized) * 100), 2)
        # videoTestAccNorm  = round((metrics.accuracy_score(videoTestingTargetsNormalized, videoTestPredictionsNormalized) * 100), 2)
        # fullTestAccNorm   = round((metrics.accuracy_score(fullTestingTargetsNormalized, fullTestPredictionsNormalized) * 100), 2)

        accResults[i, 0] = audioTestAcc
        accResults[i, 1] = videoTestAcc
        accResults[i, 2] = fullTestAcc
        # accResults[i, 3] = audioTestAccNorm
        # accResults[i, 4] = videoTestAccNorm
        # accResults[i, 5] = fullTestAccNorm

        print("\nAccuracy on audio test data for iteration " + str(i+1) + "    :   ", audioTestAcc, "%")
        print("Accuracy on video test data for iteration " + str(i+1) + "    :   ", videoTestAcc, "%")
        print("Accuracy on combined test data for iteration " + str(i+1) + " :   ", fullTestAcc, "%")
        # print("\nAccuracy on the normalized audio test data for iteration " + str(i+1) + "    :   ", audioTestAccNorm, "%")
        # print("Accuracy on the normalized video test data for iteration " + str(i+1) + "    :   ", videoTestAccNorm, "%")
        # print("Accuracy on the normalized combined test data for iteration " + str(i+1) + " :   ", fullTestAccNorm, "%")

        # calculateConfusionMatrix(audioTestingTargets, audioTestPredictions)
        # calculateConfusionMatrix(videoTestingTargets, videoTestPredictions)
        # calculateConfusionMatrix(fullTestingTargets, fullTestPredictions)

    accTotals = np.sum(accResults, axis=0)
    accTotals /= 10.0

    print("\nAverage accuracy on audio test data:      ", round((accTotals[0]), 2), "%")
    print("Average accuracy on video test data:      ", round((accTotals[1]), 2), "%")
    print("Average accuracy on combined test data:   ", round((accTotals[2]), 2), "%")
    # print("\nAverage accuracy on the normalized audio test data:      ", round((accTotals[3]), 2), "%")
    # print("Average accuracy on the normalized video test data:      ", round((accTotals[4]), 2), "%")
    # print("Average accuracy on the normalized combined test data:   ", round((accTotals[5]), 2), "%")


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
