import cv2
import numpy as np

# Load the image
image_path = "media/test_img_green.png"  # Change to your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
    exit()

# Resize for better visibility (optional)
image = cv2.resize(image, (640, 480))

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow("Trackbars")

# Function for nothing (used for trackbars)
def nothing(x):
    pass

# Create HSV trackbars
# RED
# cv2.createTrackbar("Lower H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("Lower S", "Trackbars", 34, 255, nothing)
# cv2.createTrackbar("Lower V", "Trackbars", 46, 255, nothing)
# cv2.createTrackbar("Upper H", "Trackbars", 15, 179, nothing)
# cv2.createTrackbar("Upper S", "Trackbars", 109, 255, nothing)
# cv2.createTrackbar("Upper V", "Trackbars", 200, 255, nothing)

# GREEN
cv2.createTrackbar("Lower H", "Trackbars", 32, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 12, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 12, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 82, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 162, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 200, 255, nothing)

while True:
    # Read trackbar values
    lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    # Define color range
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Detected Color", result)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()