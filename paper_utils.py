import cv2
import numpy as np

from config import config


def get_largest_contour(frame, get_main_page = True):
    # Convert the frame to grayscale
    # https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    # https://pl.wikipedia.org/wiki/Metoda_Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((10,10),np.uint8)
    cv2.morphologyEx(otsu, cv2.MORPH_DILATE, kernel)
    kernel = np.ones((10, 10), np.uint8)
    cv2.morphologyEx(otsu, cv2.MORPH_CLOSE,kernel)

    # Apply Canny edge detection
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    edges = cv2.Canny(otsu, config["canny-min"], config["canny-max"], apertureSize=config["canny-aperture"])

    # Display the Canny edge detection
    if not get_main_page:
        cv2.imshow('Canny Edge Detection', edges)

    # Find contours
    # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
    # cv2.RETR_EXTERNAL - Retrieves only the outermost contours, ignoring nested ones.
    # cv2.CHAIN_APPROX_SIMPLE - Simplifies contours by storing only essential points, reducing memory usage.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if get_main_page:
    # Find the largest contour by area
        return max(contours, key=cv2.contourArea, default=None)
    else:
    # Find contours of objects
        return [contour for contour in contours if cv2.contourArea(contour) > config["min-area"]]



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

    # print("Width: ", width, "Height: ", height)

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


def get_object_size(points):
    boundingbox_size = []
    if len(points) == 4:
        x_a = abs(points[1][0][0] - points[0][0][0])
        y_a = abs(points[1][0][1] - points[0][0][1])
        boundingbox_size.append((x_a, y_a))
        x_b = abs(points[3][0][0] - points[0][0][0])
        y_b = abs(points[0][0][1] - points[3][0][1])
        boundingbox_size.append((x_b, y_b))

    # Return a properties to compute a height and width of the object
    # Under index 0 it keeps a properties to compute width and under index 1 height
    return boundingbox_size

def get_papre_size_in_mm():
    return 210, 297

def get_measurments_real_unit(rect, points):
    a=int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
    b=int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
    paper_width_in_pixels= min(a,b)
    paper_height_in_pixels = max(a,b)

    print("--------------------------------------------------------------------------")
    print(f"paper w in px: {paper_width_in_pixels}, paper h in px: {paper_height_in_pixels}\n")

    paper_width_in_mm, paper_height_in_mm = get_papre_size_in_mm()

    print(f"paper w in mm: {paper_width_in_mm}, paper h in mm: {paper_height_in_mm}\n")

    scale_long_edge = paper_width_in_mm / paper_width_in_pixels
    scale_short_edge = paper_height_in_mm / paper_height_in_pixels

    print(f"paper scale long: {scale_long_edge}, paper scale short: {scale_short_edge}\n")

    object_in_pixels = get_object_size(points)

    object_width_in_mm = object_height_in_mm = 0
    
    if a>b:
        object_width_in_mm = ((object_in_pixels[0][0] * scale_long_edge)**2 + (object_in_pixels[0][1] * scale_short_edge)**2)**0.5
        object_height_in_mm = ((object_in_pixels[1][0] * scale_long_edge)**2 + (object_in_pixels[1][1] * scale_short_edge)**2)**0.5
    else:
        object_width_in_mm = ((object_in_pixels[0][0] * scale_short_edge)**2 + (object_in_pixels[0][1] * scale_long_edge)**2)**0.5
        object_height_in_mm = ((object_in_pixels[1][0] * scale_short_edge)**2 + (object_in_pixels[1][1] * scale_long_edge)**2)**0.5
    
    print(f"object w in px: {(object_in_pixels[0][0]**2 + object_in_pixels[0][1]**2)**0.5}, object h in px: {(object_in_pixels[1][0]**2 + object_in_pixels[1][1]**2)**0.5}\n")
    print(f"object w in mm: {object_width_in_mm}, object h in mm: {object_height_in_mm}\n")

    return object_width_in_mm, object_height_in_mm
