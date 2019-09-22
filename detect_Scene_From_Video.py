# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import dlib
import face_recognition
import cv2
import imutils
import argparse
import imutils
import os


fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

captured = False
total = 0
frames = 0
 
# open a pointer to the video file initialize the width and height of
# the frame
vs = cv2.VideoCapture("D:/scene-detector/batman_who_laughs_7.mp4")
(W, H) = (None, None)

while True:
	# grab a frame from the video
	(grabbed, frame) = vs.read()
 
	# if the frame is None, then we have reached the end of the
	# video file
	if frame is None:
		break
 
	# clone the original frame (so we can save it later), resize the
	# frame, and then apply the background subtractor
	orig = frame.copy()
	frame = imutils.resize(frame, width=600)
	mask = fgbg.apply(frame)
 
	# apply a series of erosions and dilations to eliminate noise
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
 
	# if the width and height are empty, grab the spatial dimensions
	if W is None or H is None:
		(H, W) = mask.shape[:2]
 
	# compute the percentage of the mask that is "foreground"
	p = (cv2.countNonZero(mask) / float(W * H)) * 100
    
	if p < 1.0 and not captured and frames > 200:
		# show the captured frame and update the captured bookkeeping
		# variable
		cv2.imshow("Captured", frame)
		captured = True

		# construct the path to the output frame and increment the
		# total frame counter
		filename = "{}.png".format(total)
		path = os.path.sep.join(["D:/scene-detector/OutPutNew", filename])
		total += 1

		# save the  *original, high resolution* frame to disk
		print("[INFO] saving {}".format(path))
		cv2.imwrite(path, orig)

	# otherwise, either the scene is changing or we're still in warmup
	# mode so let's wait until the scene has settled or we're finished
	# building the background model
	elif captured and p >= 10.0:
		captured = False

	# display the frame and detect if there is a key press
	cv2.imshow("Frame", frame)
	cv2.imshow("Mask", mask)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the frames counter
	frames += 1

# do a bit of cleanup
vs.release()