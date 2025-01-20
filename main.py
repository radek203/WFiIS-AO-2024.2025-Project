import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from config import config
from paper_utils import get_largest_contour, get_approx_polygon, get_warped_perspective, get_object_size, get_measurments_real_unit, get_original_coordinates, get_edge_centers

def show_frame():
    # Read a frame
    ret, frame = camera.read()
    
    # Check if the frame was read correctly
    if not ret:
        print("Error: Could not read frame.")
        return

    largest_contour = get_largest_contour(frame)

    if largest_contour is not None:
        # Approximate the contour to a polygon
        approx, rect = get_approx_polygon(largest_contour)
        
        if rect is not None:
            # Get the warped perspective
            warped = get_warped_perspective(frame, rect)
            
            # Draw the polygon and corners on the original frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            for point in approx:
                cv2.circle(frame, point[0], 10, (0, 0, 255), -1)
                
            # Detect object on warped perspective
            object_contours = get_largest_contour(warped, False)
            for object_contour in object_contours:
                if object_contour is not None:
                    approx_polygon, rect_object = get_approx_polygon(object_contour)
                    if rect_object is not None:

                        # Calculating original coordinates for object corners
                        # As parameters function gets an original an array of points and the original frame rect
                        original_points = get_original_coordinates(approx_polygon,rect)
                        cv2.drawContours(frame, [original_points], -1, (255, 0, 255), 3)
                        for point in original_points:
                            cv2.circle(frame, point[0], 5, (255, 100, 0),-1)

                        long_edge_center, short_edge_center = get_edge_centers(original_points)
                        long_edge_size, short_edge_size = get_measurments_real_unit(rect, approx_polygon, dropdown.get())
                        cv2.putText(frame, f'{long_edge_size:.0f}', tuple(long_edge_center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (44, 110, 19), 2)
                        cv2.circle(frame, tuple(long_edge_center.astype(int)), 5, (44, 110, 19), -1)
                        cv2.putText(frame, f'{short_edge_size:.0f}', tuple(short_edge_center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (256, 70, 122), 2)
                        cv2.circle(frame, tuple(short_edge_center.astype(int)), 5, (255, 70, 122), -1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(35, show_frame)


# Initialize the camera (0 is usually the default camera) 
camera = cv2.VideoCapture(config["camera"])

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Camera opened successfully.")

# Create Tkinter window
root = tk.Tk()
root.title("Mierzenie obiektów na kartce")

# Create label to display the frame
label = tk.Label(root)
label.pack()

# Create dropdown menu
options = ["A3", "A4", "A5"]
selected_option = tk.StringVar()
selected_option.set("A4") # Default option

dropdown = ttk.Combobox(root, textvariable=selected_option, values=options)
dropdown.pack()

# Start displaying frames
show_frame()

# Start Tkinter main loop
root.mainloop()
