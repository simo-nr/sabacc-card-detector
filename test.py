import cv2
import numpy as np

# Load the image
image = cv2.imread("media/shapes.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)
# cv2.imshow("Threshold", thresh)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    # Calculate the perimeter of the contour
    peri = cv2.arcLength(contour, True)
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    # Draw the approximated polygon
    cv2.polylines(image, [approx], True, (0, 255, 255), 2)  # Yellow for approximation

    # Count the number of vertices
    vertices = len(approx)

    # Detect shape based on vertices
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        # Further check if it's square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:  # Almost square
            shape = "Square"
        else:
            shape = "Rectangle"  # Not needed for your specific case
    else:
        # Detect circle by comparing area and perimeter
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        circularity = 4 * np.pi * (area / (peri * peri))
        if 0.8 <= circularity <= 1.2:  # Circularity close to 1
            shape = "Circle"
        else:
            shape = "Unknown"

    # Draw the contour and the detected shape
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green for original contour
    x, y = approx.ravel()[0], approx.ravel()[1]  # Position for text
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show the image with approximated shapes and contours
cv2.imshow("Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()