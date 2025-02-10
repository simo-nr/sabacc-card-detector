import numpy as np
import cv2

class Card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 310x500, flattened, grayed, blurred image
        self.sign = "Unknown" # positive or negative
        self.rank_img = [] # Thresholded, sized image of card's rank

        self.debug_view = [] # Debug view of card


def get_sign(card):
    """Determines the sign of the card based on overall color, 
    green for positive, red for negative"""
    
    # Array = BGR
    lower_red = np.array([50, 40, 0], dtype = "uint8")
    upper_red = np.array([100, 100, 255], dtype = "uint8")

    lower_green = np.array([30, 20, 30], dtype = "uint8")
    upper_green = np.array([100, 255, 100], dtype = "uint8")

    mask_red = cv2.inRange(card.warp, lower_red, upper_red)
    mask_green = cv2.inRange(card.warp, lower_green, upper_green)

    pixels_red = cv2.countNonZero(mask_red)
    pixels_green = cv2.countNonZero(mask_green)

    # print("Red pixels: ", pixels_red)
    # print("Green pixels: ", pixels_green)

    return "Positive" if pixels_green > pixels_red else "Negative"


