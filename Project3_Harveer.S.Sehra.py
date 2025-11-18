# This Project includes Two Major Sections 
# (1) Image Processing and (2) Componet Detection


#First Section: Using OpenCV to mask out an image of a motherboard
#Second Section: Using YOLOv11 to classify componets from a PCB

# Note: The project tasks can be handled by packages like OpenCV, Ultralytics, PyTorch, Numpy, and Pillow.

#OpenCV will be harnessed in the intial phase for precise image masking, esuring the extraction of relevant features
# from the motherboard image. Subsequently, YOLO will be emplyed to conduct componet classfication on PCB



# Step 1: Object Masking: Used for machines to identify and isolate specfici objects within images from the background.

import cv2 #Use OpenCV tools to read the image
import numpy as np

image = cv2.imread("C:/Users/Harve/Videos/Project 3 Data/Project 3 Data/motherboard_image.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply the binary thresholding

ret, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 127 is a threshold value vs the 255, white colour. It suggests what colour each pixel will be.  


cv2.imshow('Orignal Grayscale', image)
cv2.imshow('Binary Threshold', thresh_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection 

# Using method "Canny"


edges = cv2.Canny(image, threshold1=50, threshold2 = 150)


cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Countour detection

#Find countours within image 

contours, hierachy = cv2.findCountours(thresh_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copy image of countours from the orignal 
image_countours = image.copy()
cv2.drawCountours(image_countours, contours, -1, (0, 255, 0), 2)
#Green selected as colours with thickness of 2

# Display the countour image 

cv2.imshow('Countour Image', image_countours)
cv2.waitkey(0)
cv2.destroyAllWindows()


#Step 2: YOLOv11 Training 

# Train a model 

model = YOLO("yolo11n.pt") # Load the pretrained model 

# Train single most idle GPU 

Results = model.train(data="data.yaml", epochs = 100, imgsz = 1200, batch= 16, name='yolo_model')

