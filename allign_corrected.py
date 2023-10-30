# install and import above modules first
import os
import cv2
import math
import matplotlib.pyplot as pl
import pandas as pd
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

def trignometry_for_distance(a, b):
	return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
					((b[1] - a[1]) * (b[1] - a[1])))


# Initialize MTCNN for face detection
detector = MTCNN()
image = cv2.imread('cherinjaphoto.jpg')
# Detect faces in the image
results = detector.detect_faces(image)
x_f, y_f, w_f, h_f = results[0]['box']
img = image[int(y_f):int(y_f+h_f), int(x_f):int(x_f+w_f)]

left_eye_centre = results[0]['keypoints']['left_eye']
left_eye_x = left_eye_centre[0]
left_eye_y = left_eye_centre[1]
right_eye_centre = results[0]['keypoints']['right_eye']
right_eye_x = right_eye_centre[0]
right_eye_y = right_eye_centre[1]

# finding rotation direction
if left_eye_y > right_eye_y:
    print("Rotate image to clock direction")
    point_3rd = (right_eye_x, left_eye_y)
    direction = -1  # rotate image direction to clock
else:
    print("Rotate to inverse clock direction")
    point_3rd = (left_eye_x, right_eye_y)
    direction = 1  # rotate inverse direction of clock

#cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
a = trignometry_for_distance(left_eye_centre,
                             point_3rd)
b = trignometry_for_distance(right_eye_centre,
                             point_3rd)
c = trignometry_for_distance(right_eye_centre,
                             left_eye_centre)
cos_a = (b * b + c * c - a * a) / (2 * b * c)
angle = (np.arccos(cos_a) * 180) / math.pi
print(angle)

if direction == -1:
    angle = 90 - angle
else:
    angle = angle

# rotate image
new_img = Image.fromarray(image)
new_img = np.array(new_img.rotate(direction * angle))
cv2.imshow('image',new_img)
cv2.imwrite('new_alligned.jpg',new_img)
#cv2.waitKey(delay= 10000)




