import cv2

from config import config
from paper_utils import get_largest_contour, get_approx_polygon, get_warped_perspective


def main():
    # Initialize the camera (0 is usually the default camera)
    camera = cv2.VideoCapture(config["camera"])

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not access the camera.")
    else:
        print("Camera opened successfully.")

    # Capture frames from the camera
    while True:
        # Read a frame
        ret, frame = camera.read()

        # Check if the frame was read correctly
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        largest_contour = get_largest_contour(frame)

        if largest_contour is not None:
            # Approximate the contour to a polygon
            approx, rect = get_approx_polygon(largest_contour)
            if rect is not None:
                warped = get_warped_perspective(frame, rect)

                # Draw the polygon and corners on the original frame
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                for point in approx:
                    cv2.circle(frame, point[0], 10, (0, 0, 255), -1)

                # Show the original frame with the polygon and corners
                cv2.imshow('Original Frame with Polygon', frame)
                # Show the warped perspective
                cv2.imshow('Warped Perspective', warped)

        # Exit when the user presses the 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
