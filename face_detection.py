# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2


# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
for i in range (0,100000):

	# reads frames from a camera
	ret, img = cap.read()

	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Detects eyes of different sizes in the input image

		#To draw a rectangle in eyes


	# Display an image in a window
	cv2.imshow('img',img)

	# Wait for Esc key to stop
	k = cv2.waitKey(0) & 0xFF
	if k == 32:
		break
	cv2.imwrite(str(i)+"im.jpg",img)

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
