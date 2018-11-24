import numpy as np

# I grabbed all of the ones above the 0.0065 threshold

# this yielded 68.7% accuracy on an audio run
importantAudioFeatures = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 22, 25, 28, 30, 32, 34, 36, 37, 39, 40, 42, 44, 46, 48, 49, 53, 56, 66, 69, 76, 104, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 134])

# this yielded 62.53% accuracy on an image run
importantImageFeatures = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36, 41, 46, 48, 49, 50, 51, 89, 98, 99, 100, 101, 102, 107, 112, 113, 114, 116, 117, 118, 119, 121, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134])

# this yielded 62.56% accuracy on a combined run
importantCombinedFeatures = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 28, 30, 34, 35, 36, 37, 40, 42, 46, 48, 49, 50, 100, 108, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134])

# these are the elements that all 3 have in common
sharedImportantFeatures = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 11, 12, 13, 14, 15, 16, 17, 19, 34, 36, 46, 48, 49, 117, 118, 119, 121, 124, 125, 126, 128, 134])