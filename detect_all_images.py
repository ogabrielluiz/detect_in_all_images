# USAGE
# Run detect_all_images.py. It gets images from /images folder and saves them to blurred_faces folder as 'blurred.IMAGENAME'

# import the necessary packages
import numpy as np
import cv2
import os

if not os.path.exists('images'):
	os.makedirs('images')
	print('[INFO] images Folder Created.')
if not os.path.exists('blurred_faces'):
	os.makedirs('blurred_faces')
	print('[INFO] blurred_faces Folder Created.')

#args : model, confidence(this one can be changed to allow for worse or better detections), image, prototxt
args = {
	"image":"",
	"model":"res10_300x300_ssd_iter_140000.caffemodel",
	"prototxt":"deploy.prototxt.txt",
	"confidence": 0.5
}

# Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


print("[INFO] computing object detections...")
# Loop  through each image and run the detection
for root,dirs,files in os.walk('images'):
	for i in files:
		# Load the input image and construct an input blob for the image
		# by resizing to a fixed 300x300 pixels and then normalizing it
		args["image"] = i
		image = cv2.imread("images/" + args["image"])
		result_image= image.copy()
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		# Pass the blob through the network and obtain the detections and
		# predictions
		
		net.setInput(blob)
		detections = net.forward()

		# Loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]
			
			# Filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				
				just_face = image[startY:endY, startX:endX]
				# apply a gaussian blur on this new recangle image
				just_face = cv2.GaussianBlur(just_face,(23, 23), 30)
				# merge this blurry rectangle to our final image
				result_image[startY:startY+just_face.shape[0], startX:startX+just_face.shape[1]] = just_face
				blurred_name = 'blurred.'+ args['image']
				cv2.imwrite('blurred_faces/'+ blurred_name , result_image)
				# Uncomment to see images one by one
				#cv2.imshow("Output", result_image)
		print("Saving " + blurred_name + " in blurred_faces folder")
		# Uncomment this too to allow you see each resulting image
		#cv2.waitKey(0)


