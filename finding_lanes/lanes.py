import cv2
import numpy as np


"""
Summary: 
Convert road image into greyscale, reduce noise using GaussianBlur and perform Canny edge detection
Identify lane inside a gradient image using Hough Transform
Take detected lines and place on a black image with same dimension as original image and blend the two in order to place detected lines onto original road image
"""


def canny(image):
    '''
    Takes a multi-dimensional numpy array image and perform canny detection
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to greyscales
    blur = cv2.GaussianBlur(gray, (5,5), 0) # apply blur to reduce noise in image
    canny = cv2.Canny(blur, 50, 150) # apply Canny function to outline strongest gradient (change between pixel value 0 to 255)
    return canny


def region_of_interest(image):
    '''
    Return enlosed region of interest
    '''
    height = image.shape[0] # Set height taken from 2D array of tuple (row, column)
    polygons = np.array([[(200, height), (1100, height), (550, 250)]]) # Polygon with cordinates representing region of interest (ROI) that will be applied ontop of mask
    mask = np.zeros_like(image) # Black image with same dimensions as image
    cv2.fillPoly(mask, polygons, 255) # Apply ROI onto mask and set to color to white
    masked_image = cv2.bitwise_and(image, mask) # Preform bitwise_and between image and mask to highlight ROI and mask everything outside ROI 
    return masked_image


def display_lines(image, lines):
    '''
    Draw detected lines on a black image
    '''
    line_image = np.zeros_like(image)
    if lines is not None:
        for  x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_parameters):
    '''
    Create coordinates given slope and y-intercept
    '''
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    # y = mx + b --rewritten--> x = (y - b)/m
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    '''
    Take detected lane lines and average them out to return one singular left and right lane line
    '''
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # Returns array of [slope, y-intercept] given points of line
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0: # If slope is negative, points belong to the left lane, otherwise belong toright lane 
            left_fit.append((slope, y_intercept))
        else:
            right_fit.append((slope, y_intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


## Uncomment to perform lane detection on image. Comment out the video code below
# image = cv2.imread('test_image.jpg') # read the image as a multi-dimensional numpy array
# lane_image = np.copy(image) # make copy of image array
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # Apply Hough Transform with 2px and 1 radian degree precision with 100 as threshold for minimum of intersection
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # Blend original image and detected lane lines

# cv2.imshow('result', combo_image) # render image
# cv2.waitKey(0) # display result window until key press


# Code to perform lane detection on video. Comment out image lane detection code above
cap = cv2.VideoCapture('test2.mp4')
while cap.isOpened():
    _, frame = cap.read() # Get frames of video
    canny_image = canny(frame) # perform canny edge detection on video fram
    cropped_image = region_of_interest(canny_image) # add region of interest 
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # Apply Hough Transform with 2px and 1 radian degree precision with 100 as threshold for minimum of intersection
    averaged_lines = average_slope_intercept(frame, lines) # Get average of lines detected
    line_image = display_lines(frame, averaged_lines) # Average the lines into one smoother line
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # Blend original image and detected left and right lane lines
    cv2.imshow('result', combo_image) # render image
    if cv2.waitKey(1) & 0xFF == ord ('q'):  # wait 1 millisecond between frames. press keyboard Q to quit
        break
cap.release()
cv2.destroyAllWindows