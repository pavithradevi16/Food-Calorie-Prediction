import numpy as np
import cv2
from create_feature import *
from calorie_calc import *
import csv

svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383)


def training():
    feature_mat = []
    response = []
    max_length = 0  # Track the maximum length of feature vectors

    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for i in range(1, 21):
            print("/Users/Ajaysaravanan/Documents/ML-Project/images/All_Images/" + str(j) + "_" + str(i) + ".jpg")
            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(
                "/Users/Ajaysaravanan/Documents/ML-Project/images/All_Images/" + str(j) + "_" + str(i) + ".jpg")

            # Track the maximum length of feature vectors
            max_length = max(max_length, len(fea))

            feature_mat.append(fea)
            response.append(int(j))

    # Pad feature vectors with zeros to make them consistent
    for i in range(len(feature_mat)):
        padding_length = max_length - len(feature_mat[i])
        feature_mat[i] = np.pad(feature_mat[i], (0, padding_length), mode='constant')

    trainData = np.float32(feature_mat).reshape(-1, max_length)
    responses = np.array(response, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    trainData = cv2.ml.TrainData_create(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.train(trainData)
    svm.save('svm_data.dat')


def testing():
    svm_model = cv2.ml.SVM_load('svm_data.dat')
    feature_mat = []
    response = []
    image_names = []
    pix_cm = []
    fruit_contours = []
    fruit_areas = []
    fruit_volumes = []
    fruit_mass = []
    fruit_calories = []
    skin_areas = []
    fruit_calories_100grams = []

    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for i in range(21, 26):
            img_path = ("/Users/Ajaysaravanan/Documents/ML-Project/images/Test_Images/" + str(j) + "_" + str(i) + ".jpg")
            print(img_path)
            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
            if fea is None:
                continue
            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)
            response.append([int(j)])
            image_names.append(img_path)

    if len(feature_mat) == 0:
        print("No valid test images found.")
        return

    max_length = len(feature_mat[0])  # Get the maximum length of feature vectors

    # Pad feature vectors with zeros to make them consistent
    for i in range(len(feature_mat)):
        padding_length = max_length - len(feature_mat[i])
        feature_mat[i] = np.pad(feature_mat[i], (0, padding_length), mode='constant')

    testData = np.float32(feature_mat).reshape(-1, max_length)
    responses = np.array(response, dtype=np.int32)

    _, result = svm_model.predict(testData)
    result = result.flatten()  # Flatten the result array to make it iterable

    # Calculate volume, mass, and calories for each fruit
    for i in range(len(result)):
        volume = getVolume(result[i], fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i])
        mass, calories, calories_100grams = getCalorie(result[i], volume)
        fruit_volumes.append(volume)
        fruit_mass.append(mass)
        fruit_calories.append(calories)
        fruit_calories_100grams.append(calories_100grams)

    # Save the results to a CSV file
    with open('test_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        data = ["Image name", "Desired response", "Output label", "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
        writer.writerow(data)
        for i in range(0, len(result)):
            if fruit_volumes[i] is None:
                data = [str(image_names[i]), str(responses[i][0]), str(result[i]), "--", "--", "--", str(fruit_calories_100grams[i])]
            else:
                data = [str(image_names[i]), str(responses[i][0]), str(result[i]), str(fruit_volumes[i]), str(fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
            writer.writerow(data)


if __name__ == '__main__':
    training()
    testing()
