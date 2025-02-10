import cv2
import numpy as np

image = cv2.imread('test2.png')

lower_red = np.array([50, 40, 0], dtype = "uint8")
upper_red = np.array([100, 100, 255], dtype = "uint8")

lower_green = np.array([30, 20, 30], dtype = "uint8")
upper_green = np.array([100, 255, 100], dtype = "uint8")

lower_blue = np.array([0, 0, 0], dtype = "uint8")
upper_blue = np.array([255, 30, 30], dtype = "uint8")

mask_red = cv2.inRange(image, lower_red, upper_red)
mask_green = cv2.inRange(image, lower_green, upper_green)
mask_blue = cv2.inRange(image, lower_blue, upper_blue)

pixels_red = cv2.countNonZero(mask_red)
pixels_green = cv2.countNonZero(mask_green)
pixels_blue = cv2.countNonZero(mask_blue)

print("Red pixels: ", pixels_red)
print("Green pixels: ", pixels_green)
print("Blue pixels: ", pixels_blue)

# combine the masks
# combined_mask = cv2.bitwise_or(mask_red, mask_green)
# combined_mask = cv2.bitwise_or(combined_mask, mask_blue)

detected_output = cv2.bitwise_and(image, image, mask = mask_red)

cv2.imshow("RGB color detection", detected_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
