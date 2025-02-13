import cv2
import numpy as np

image = cv2.imread('media/test_img_6.png')

# lower_red = np.array([50, 40, 0], dtype = "uint8")
# upper_red = np.array([100, 100, 255], dtype = "uint8")

# lower_green = np.array([30, 20, 30], dtype = "uint8")
# upper_green = np.array([100, 255, 100], dtype = "uint8")

# lower_blue = np.array([0, 0, 0], dtype = "uint8")
# upper_blue = np.array([255, 30, 30], dtype = "uint8")



lower_hsv_red = np.array([0, 34, 46])
upper_hsv_red = np.array([180, 178, 138])

lower_hsv_green = np.array([32, 12, 12])
upper_hsv_green = np.array([82, 162, 129])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask_red = cv2.inRange(hsv, lower_hsv_red, upper_hsv_red)
mask_green = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)
# mask_blue = cv2.inRange(image, lower_blue, upper_blue)

pixels_red = cv2.countNonZero(mask_red)
pixels_green = cv2.countNonZero(mask_green)
# pixels_blue = cv2.countNonZero(mask_blue)

print("Red pixels: ", pixels_red)
print("Green pixels: ", pixels_green)
# print("Blue pixels: ", pixels_blue)

# combine the masks
# combined_mask = cv2.bitwise_or(mask_red, mask_green)
# combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
detected_output = cv2.bitwise_and(image, image, mask = mask_red)
detected_output_green = cv2.bitwise_and(image, image, mask = mask_green)

cv2.imshow("RGB color detection", detected_output)
cv2.imshow("RGB color detection green", detected_output_green)

cv2.imshow("Red Mask", mask_red)
cv2.imshow("Green Mask", mask_green)

cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image
# image = cv2.imread("media/test_img_1.png")

# # Convert to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define HSV range for red color
# # lower_red = np.array([0, 120, 70])     # Lower bound of red in HSV
# # upper_red = np.array([10, 255, 255])   # Upper bound of red in HSV

# lower_red = np.array([170, 100, 80]) 
# # upper_red = np.array([175, 184, 138])  # (H, S, V) in OpenCV
# upper_red = np.array([180, 255, 200])

# # Create a mask
# mask = cv2.inRange(hsv, lower_red, upper_red)

# # Apply the mask to the original image
# result = cv2.bitwise_and(image, image, mask=mask)

# # Show images
# # cv2.imshow("Original", image)
# cv2.imshow("Red Mask", mask)
# cv2.imshow("Filtered Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_sign(card):
    """Determines the sign of the card based on overall color, 
    green for positive, red for negative"""
    
    # Array = BGR
    # lower_red = np.array([50, 40, 0], dtype = "uint8")
    # upper_red = np.array([100, 100, 255], dtype = "uint8")

    # lower_green = np.array([30, 20, 30], dtype = "uint8")
    # upper_green = np.array([100, 255, 100], dtype = "uint8")

    lower_hsv_red = np.array([0, 34, 46])
    upper_hsv_red = np.array([180, 178, 138])

    lower_hsv_green = np.array([32, 12, 12])
    upper_hsv_green = np.array([82, 162, 129])

    hsv = cv2.cvtColor(card.warp, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_hsv_red, upper_hsv_red)
    mask_green = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)

    pixels_red = cv2.countNonZero(mask_red)
    pixels_green = cv2.countNonZero(mask_green)

    # print("Red pixels: ", pixels_red)
    # print("Green pixels: ", pixels_green)

    return "Positive" if pixels_green > pixels_red else "Negative"