import cv2
import numpy as np
import sys

def getShapeFeatures(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        moments = cv2.moments(contours[0])
        hu = cv2.HuMoments(moments)
        feature = []
        for i in hu:
            feature.append(i[0])
        M = max(feature)
        m = min(feature)
        feature = [(x - M - m) / (M - m) for x in feature]
        mean = np.mean(feature)
        dev = np.std(feature)
        feature = [(x - mean) / dev for x in feature]
        return feature
    else:
        return []

def training():
    feature_mat = []
    for j in range(1, 5):
        for i in range(1, 71):
            img = cv2.imread("/Users/Ajaysaravanan/Documents/ML-Project/images/All_Images/" + str(j) + "_" + str(i) + ".jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(img, 80, 255)
            img1 = cv2.bitwise_and(img, img, mask=mask)
            features = getShapeFeatures(img1)
            if len(features) > 0:
                feature_mat.append(features)

    trainData = np.array(feature_mat, dtype=np.float32)
    trainData = trainData.reshape(-1, len(feature_mat[0]))

    # Rest of your training code

if __name__ == '__main__':
    training()
