# Description: This program attempts to build
# an end-to-end pipeline that allows the end user
# to capture an image and upload it to the pipeline
# where it attempts to apply Optical Character Recognition
# techniques to convert possibly illegible handwritten documents
# to a readable, text generated format

import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import os
import requests
import pytesseract
from PIL import Image as im
from scipy.ndimage import interpolation as inter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Applies adaptive thresholding to a passed img
# Better to use as user will take image using their phone,
# prone to uneven lighting
def apply_adaptiveTresh(img):
    # Stores the binary image
    binaryImg = np.array(img)
    threshImg = cv2.adaptiveThreshold(binaryImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return threshImg


# Applies binarisation to an image
def convertImg_binary(img):
    binary_imgCopy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_imgCopy = cv2.medianBlur(binary_imgCopy, 5)
    # width, height = np.size(img)
    # pixels = np.array(imgCopy.convert('1').getdata(), np.uint8)
    # binary_imgCopy = (1 - (pixels.reshape(height, width) / 255))
    return binary_imgCopy


# Calculates & returns the angle of a skewed image
def getImg_angle(thresholdImg):
    imgCopy = thresholdImg

    # Converts 1D array into 2D by stacking as columns
    img_coordinates = np.column_stack(np.where(imgCopy > 0))
    img_angle = cv2.minAreaRect(img_coordinates)[-1]

    # Inverting the angle
    if img_angle > -45:
        img_angle = -img_angle
    else:
        img_angle = -(90 + img_angle)
    return img_angle


# Deskews an image and returns it
def deskew_img(img, img_angle):
    (height, width) = img.shape[:2]
    img_midpoint = (width // 2, height // 2)
    midpoint = cv2.getRotationMatrix2D(img_midpoint,
                                       img_angle, 1.0)

    # Applying affine transformation to ensure all
    # parallel lines in base image are maintained in
    # the newly deskewed image 
    img_deskewed = cv2.warpAffine(img, midpoint, (width, height),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img_deskewed


def remove_noise(img):
    imgCopy = img

    # Requires passed image to be gray
    denoised_img = cv2.fastNlMeansDenoising(imgCopy,
                                            None, 30.0, 7, 21)

    return denoised_img


# Handwriting is not uniform, people have
# different writing styles, meaning width
# of characters will be different, must
# apply errorion to address this
def erode_img(img):
    imgCopy = img
    # Creating a 5x5 matrix for the kernel
    kernel = np.ones((5, 5), np.uint8)

    # Applying morphological transformations
    # Tried to apply erosion after dilation but noticed
    eroded_img = cv2.erode(imgCopy, kernel, iterations=1)
    imgCopy = cv2.dilate(eroded_img, kernel, iterations=1)

    return imgCopy

# Applying segmentation with the below functions

# Finds and joins countours of characters within an image
def getImg_countour(thresholdImg):
    imgCopy = thresholdImg

    # Finds boundary points of characters in image
    # Foreground text must be of same colour as
    # this is how these points are joined
    # RETR_LIST returns all contours found
    char_countour, hierarchy = cv2.findContours(imgCopy,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Continuously iterates to find countours
    # and connects them to form boxes representing
    # the segmented characters
    for countour in char_countour:
        xCoord, yCoord, width, height = cv2.boundingRect(countour)
        if height > 8:
            cv2.rectangle(imgCopy, (xCoord, yCoord),
            (xCoord+width, yCoord+height), (0,255,0),2)

    return imgCopy

# Generates a projection histogram of an image
def generate_projectionHist(thresholdImg, contouredImg, img):
    imgCopy_thresh = thresholdImg
    imgCopy_contour = contouredImg
    height, width = imgCopy_thresh.shape

    imgPixls_vertical = np.sum(imgCopy_thresh, axis=0)

    # Normalizing to avoid gradients exploding
    # due to pixels being of too high range
    norm = imgPixls_vertical/255

    # Returns an array with the same type and size of the image
    # to include zeros
    black_img = np.zeros_like(imgCopy_thresh)

    # Generate the histogram
    # Goes through each column and will create a line
    # in accordance to the number of black pixels as shown
    # above to produce vertical projections
    for index, value in enumerate(norm):
        # draws line between each point
        cv2.line(black_img, (index, 0),
        (index,height-int(value)), 
        (255,255,255),1)
    
    # Combine the generated histogram with
    # the image containing segmented characters
    # Images must be of same size to concatenate
    projection_img = cv2.vconcat(
        [img, cv2.cvtColor(black_img,
        cv2.COLOR_BGR2RGB)]
    )

    return projection_img

# To adhere to the spec proposal for this assignment,
# an examiner should be able to user their phone to 
# take an image and upload it to the application.
def capture_image():
    # Access the VideoCapture object
    user_camera = cv2.VideoCapture(0)
    loop_count = 0

    # Must constantly be running until user has taken image
    while True:
        valid, frame = user_camera.read()
        cv2.imshow("UserCamera", frame)

        # 32 represents the spacebar
        exit_key = cv2.waitKey(32)

        # If spacebar is pressed exit the window
        if exit_key == (32):
            break

        cv2.imwrite("C:/" + "examPaper{}.jpg".format(loop_count), frame)
        saved_img = frame

    # Destroying all instances of webcam
    user_camera.release()
    cv2.destroyAllWindows()

    return saved_img

# Saves passed text to a text file
def generate_textFile(fileName, filePath, textToSave):
    full_name = os.path.join(filePath, fileName + '.txt')
    saved_file = open(full_name, 'w')
    saved_file.write(textToSave)
    saved_file.close()

# Entry point to pipeline
def main():
    # Retrieving image
    captured_img = capture_image() # Allow user to take picture of paper

    # Preprocessing
    binary_img = convertImg_binary(captured_img) # Apply binarisation
    thresh_img = apply_adaptiveTresh(binary_img) # Apply adaptive threshold
    noNoise_img = remove_noise(thresh_img) # Remove noise 
    img_eroded = erode_img(noNoise_img) # Apply erosion
    img_deskewed = deskew_img(img_eroded, (getImg_angle(img_eroded))) # Deskew image

    # Attempted segmentation
    projection_hist = generate_projectionHist(img_deskewed, 
    (getImg_countour(img_deskewed)), captured_img)

    # Displaying results
    plt.imshow(projection_hist, cmap='gray')
    plt.show()

    # Converting to text and saving locally
    saved_text = pytesseract.image_to_string(captured_img)
    generate_textFile('studentPaper','C:/', saved_text)


main()



