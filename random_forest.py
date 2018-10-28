import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_FILE = "data/column_3C_weka.csv"
FEATURES = 6
TEST_SIZE = 0.2


def main():

    features_train, features_test, labels_train, labels_test = prepareDataset(DATA_FILE)

    #Scale / Normalize the features
    sc = StandardScaler()
    features_train = sc.fit_transform(features_train)
    features_test = sc.transform(features_test)

    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(features_train, labels_train)
    predictions = classifier.predict(features_test)

    print(confusion_matrix(labels_test,predictions))
    #print(classification_report(labels_test,predictions))
    print(accuracy_score(labels_test, predictions))


''' '''
def prepareDataset(fileName):
    dataset = pd.read_csv(fileName)
    print "\nColumns: ", list(dataset), "\n"

    #Seperates the dataset into attributes and labels (assuming last column is label)
    X = dataset.iloc[:, 0:FEATURES].values   #features
    y = dataset.iloc[:, FEATURES].values     #labels

    #Seperates into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

    return features_train, features_test, labels_train, labels_test


'''Displays a confusion matrix of the actual versus predicted values '''
def displayConfusionMatrix(actual, predicted):
    cm = np.zeros((2, 2), int)
    for a, p in zip(actual, predicted):
        cm[a,p] += 1

    df_cm = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()


main()
