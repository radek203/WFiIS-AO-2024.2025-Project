import cv2
import numpy as np

def get_largest_contour(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    edges = cv2.Canny(otsu, 50, 100, apertureSize=5)

    # Display the Canny edge detection
    cv2.imshow('Canny Edge Detection', edges)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    return max(contours, key=cv2.contourArea, default=None)


def get_approx_polygon(largest_contour):
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check if the approximated polygon has (hasn't) 4 points
    if len(approx) != 4:
        return None, None

    # Sort the points to ensure correct order: top-left, top-right, bottom-right, bottom-left
    points = np.array([point[0] for point in approx], dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top-left
    rect[2] = points[np.argmax(s)]  # Bottom-right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top-right
    rect[3] = points[np.argmax(diff)]  # Bottom-left
    return approx, rect


def get_warped_perspective(frame, rect):
    # Define the width and height of the new perspective image
    width = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
    height = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))

    print("Width: ", width, "Height: ", height)

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective warp
    return cv2.warpPerspective(frame, matrix, (width, height))
