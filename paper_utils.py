import cv2
import numpy as np

from config import config


def get_largest_contour(frame):
    # Convert the frame to grayscale
    # https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    # https://pl.wikipedia.org/wiki/Metoda_Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    edges = cv2.Canny(otsu, config["canny-min"], config["canny-max"], apertureSize=config["canny-aperture"])

    # Display the Canny edge detection
    cv2.imshow('Canny Edge Detection', edges)

    # Find contours
    # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
    # cv2.RETR_EXTERNAL - Retrieves only the outermost contours, ignoring nested ones.
    # cv2.CHAIN_APPROX_SIMPLE - Simplifies contours by storing only essential points, reducing memory usage.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    return max(contours, key=cv2.contourArea, default=None)


def get_approx_polygon(largest_contour):
    # epsilon is the maximum distance from the contour to the approximated contour
    # We multiply the epsilon by the arc length of the contour to make it relative to the size of the contour
    epsilon = config["approx-epsilon"] * cv2.arcLength(largest_contour, True)
    # https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check if the approximated polygon has (hasn't) 4 points
    if len(approx) != 4:
        return None, None

    # Sort the points to ensure correct order: top-left, top-right, bottom-right, bottom-left
    points = np.array([point[0] for point in approx], dtype="float32") # All points
    rect = np.zeros((4, 2), dtype="float32") # All points in correct order

    s = np.sum(points, axis=1)
    rect[0] = points[np.argmin(s)]  # Top-left, Smallest sum of x and y
    rect[2] = points[np.argmax(s)]  # Bottom-right, Largest sum of x and y

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top-right, Smallest difference of x and y
    rect[3] = points[np.argmax(diff)]  # Bottom-left, Largest difference of x and y
    return approx, rect


def get_warped_perspective(frame, rect):
    # Define the width and height of the new perspective image
    width = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0]))) # Largest norm distance between points (bottom-right and bottom-left, top-right and top-left)
    height = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3]))) # Largest norm distance between points (top-right and bottom-right, top-left and bottom-left)

    print("Width: ", width, "Height: ", height)

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective warp
    return cv2.warpPerspective(frame, matrix, (width, height))
