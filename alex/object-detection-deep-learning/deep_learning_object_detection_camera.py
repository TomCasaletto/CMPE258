# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from re import S
import numpy as np
import argparse
import cv2
import tensorflow as tf
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
from tensorflow.core.protobuf.meta_graph_pb2 import AssetFileDef
from tensorflow.python.framework.tensor_shape import as_dimension
from tensorflow.tools.docs.doc_controls import set_deprecated

def detectImage(imagePath,model,prototxt,confidence1,DNN):

	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# ap.add_argument("-p", "--prototxt", required=True,
	# 	help="path to Caffe 'deploy' prototxt file")
	# ap.add_argument("-m", "--model", required=True,
	# 	help="path to Caffe pre-trained model")
	# ap.add_argument("-c", "--confidence", type=float, default=0.2,
	# 	help="minimum probability to filter weak detections")
	# args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	if DNN == "CAFFE":
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		net = cv2.dnn.readNetFromCaffe(prototxt, model)
	else:
		CLASSES = ["background"]
		prototxtTemp = model.split('/')
		prototxtTemp = '/'.join(prototxtTemp[:-1])
		prototxt =  prototxtTemp + '/graph.pbtxt'
		# cv2.dnn.writeTextGraph(model[:-4], prototxt)
		# net = cv2.dnn.readNetFromTensorflow(model[:-4])
		# net = tf.keras.models.load_model(model)
		# net = cv2.d  nn.readNet(model)



	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	# print("[INFO] loading model...")
	# net = cv2.dnn.readNetFromCaffe(prototxt, model)

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)

	# created a *threaded *video stream, allow the camera senor to warmup,
	# and start the FPS counter
	print("[INFO] sampling THREADED frames from webcam...")
	vs = WebcamVideoStream(src=0).start()
	fps = FPS().start()

	while fps._numFrames < 1000:
		frame = vs.read()
		image= imutils.resize(frame, width=400)
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		# print("[INFO] computing object detections...")
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > confidence1:
				# extract the index of the class label from the `detections`,
				# then compute the (x, y)-coordinates of the bounding box for
				# the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# display the prediction
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				# print("[INFO] {}".format(label))
				cv2.rectangle(image, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		# show the output image
		cv2.imshow("Output", image)
		cv2.waitKey(1)
		fps.update()
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()





mypath = '/Users/alexdodd/Documents/CMPE 258 Deep Learning/object-detection-deep-learning/'

detectImage(mypath + 'images/', mypath + 'MobileNetSSD_deploy.caffemodel',  \
	mypath + 'MobileNetSSD_deploy.prototxt.txt', .6,'CAFFE')         
