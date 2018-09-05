#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
from scipy import io
import numpy as np
import random
import imutils
import cv2

model_output = "model/"
facePath = "dataset/faces_aligned"
faces_min = 8
face_size = (47, 62)
test_size = 0.15

def load_sunplusit_faces(datasetPath, min_faces=10, face_size=face_size, equal_samples=True,
	test_size=test_size, seed=42):
	imagePaths = sorted(list(paths.list_images(datasetPath)))

	# set the random seed, then initialize the data matrix and labels
	random.seed(seed)
	data = []
	labels = []

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		print("Image",imagePath)
		# load the image and convert it to grayscale
		face = cv2.imread(imagePath)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		face = cv2.resize(face, face_size)

		# update the data matrix and associated labels
		data.append(face)

		labels.append(imagePath.split("/")[-2])

	# convert the data matrix and labels list to a NumPy array
	data = np.array(data)
	labels = np.array(labels)

	# # check to see if equal samples for each face should be used
	if equal_samples:
		# initialize the list of sampled indexes
		sampledIdxs = []

		# 傳回該label位於labels的第幾個index，格式為array，如[0 3 6 9]
		for label in np.unique(labels):
			# len(labelIdxs)為該label的數量，若數量大於要求的最小數量min_faces
			labelIdxs = np.where(labels == label)[0]

			# 則從labelIdxs中取得不重複,數量為min_faces
			if len(labelIdxs) >= min_faces:
				# randomly sample the indexes for the current label, keeping only minumum
				# supplied amount, then update the list of sampled idnexes
				labelIdxs = random.sample(list(labelIdxs), min_faces)
				sampledIdxs.extend(labelIdxs)

		# use the sampled indexes to select the appropriate data points and labels
		random.shuffle(sampledIdxs)
		data = data[sampledIdxs]
		labels = labels[sampledIdxs]

	# compute the training and testing split index
	idxs = range(0, len(data))
	random.shuffle(list(idxs))
	split = int(len(idxs) * (1.0 - test_size))

	# split the data into training and testing segments
	(trainData, testData) = (data[:split], data[split:])
	(trainLabels, testLabels) = (labels[:split], labels[split:])

	# create the training and testing bunches
	training = Bunch(name="training", data=trainData, target=trainLabels)
	testing = Bunch(name="testing", data=testData, target=testLabels)

	# return a tuple of the training, testing bunches, and original labels
	return (training, testing, labels) 
	
(training, testing, names) = load_sunplusit_faces(facePath, min_faces=faces_min, test_size=test_size)

le = LabelEncoder()
le.fit_transform(training.target)

#recognizer = cv2.face.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=8, grid_y=8)
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

print("[INFO] training face recognizer...")
recognizer.train(training.data, le.transform(training.target))

predictions = []
confidence = []
# loop over the test data
for i in range(0, len(testing.data)):
	print("{} of {}".format(str(i), str(len(testing.data))))
	# classify the face and update the list of predictions and confidence scores
	(prediction, conf) = recognizer.predict(testing.data[i])
	predictions.append(prediction)
	confidence.append(conf)
	
# show the classification report
print(classification_report(le.transform(testing.target), predictions, target_names=np.unique(names)))


np.savetxt(model_output+'target.out', training.target, delimiter=',', fmt="%s") 
print("Exporting model...")
recognizer.write(model_output+"boss.yaml")


