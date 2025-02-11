import numpy as np
import cv2
from collections import Counter

IGNORE_VALUE = -11

font = cv2.FONT_HERSHEY_SIMPLEX

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
        self.rank = "Unknown" # Rank of card
        self.suit = "Unknown" # Suit of card
        
        self.debug_view = [] # Debug view of card

    def get_rank(self):
        """return the most frequent element in rank_img list"""
        # counter = Counter([x for x in self.rank_list if x != IGNORE_VALUE])
        counter = Counter(self.rank_list)
        most_common_elem = counter.most_common(1)[0][0]
        return str(most_common_elem)


def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    card = Card()

    card.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    card.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    card.width, card.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]

    # Warp card into 310x500 flattened image using perspective transform
    card.warp = flatten(image, pts, w, h)

    card.sign = get_sign(card)

    # Place the sign of the card on top of the debug view
    card.debug_view = card.warp.copy()
    cv2.putText(card.debug_view, card.sign, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    card.rank, card.suit = get_rank_and_suit(card)

    return card

def get_rank_and_suit(card):
    # Define the coordinates for the region of interest (ROI)
    x_start, y_start = 55, 115  # Starting coordinates (top-left corner)
    x_end, y_end = 255, 385     # Ending coordinates (bottom-right corner)
    
    # Crop and treshold the rank image from the card
    rank_img = card.warp[y_start:y_end, x_start:x_end]
    thresh_level = 200
    _, card.rank_img = cv2.threshold(rank_img, thresh_level, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(card.rank_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Get the number of distinct shapes
    num_shapes = len(contours)
    # if num_shapes != 2 and num_shapes != 3:
    #     print(f"Number of distinct shapes: {num_shapes}")
    #     cv2.imwrite(f"wrong_rank_{num_shapes}.png", image_with_contours)

    # cv2.imshow("Contours", image_with_contours)

    # Determine suit
    # Iterate through contours
    for contour in contours:
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Draw the approximated polygon
        cv2.polylines(gray, [approx], True, (0, 255, 255), 2)  # Yellow for approximation

        # Count the number of vertices
        vertices = len(approx)

        # Detect shape based on vertices
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle" # in case of weird stuff
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

    return str(num_shapes), shape

def flatten(image, pts, w, h):
    """Flattens an image of a card into a top-down 310x500 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    # top left smalles sum
    # bottom right largest sum
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # top right smallest diff
    # bottom left largest diff
    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp

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


