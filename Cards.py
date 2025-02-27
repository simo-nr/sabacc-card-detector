import numpy as np
import cv2
import math
from collections import Counter

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



def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def orientation(p1, p2):
    """Returns orientation of p2 relative to p1."""
    dx = abs(p1[0] - p2[0])  # Horizontal distance
    dy = abs(p1[1] - p2[1])  # Vertical distance

    if dy > dx:
        if p1[1] < p2[1]:
            return "above"
        else:
            return "below"
    else:
        if p1[0] < p2[0]:
            return "left"
        else:
            return "right"


def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    card = Card()

    card.contour = contour

    # Find perimeter of card and use it to approximate corner points
    contour = cv2.convexHull(contour)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.003 * peri, True)
    pts = np.float32(approx)
    card.corner_pts = pts

    # draw approx on image
    cv2.polylines(image, [approx], True, (0, 255, 0), 2)
    # print(len(pts))

    # find 4 longest lines in approx
    longest_edges = find_longest_edges(approx, 4)

    # draw the 4 longest lines on the image
    for edge in longest_edges:
        cv2.line(image, edge[0], edge[1], (0, 100, 255), 2)

    # extend the lines
    extended_lines = []
    for pt1, pt2 in longest_edges:
        p1, p2 = extend_line(pt1, pt2)
        extended_lines.append((p1, p2))
        # draw_line_from_equation(image, m, b)
        cv2.line(image, p1, p2, (255, 0, 0), 1)

    # find the intersection of the lines
    for i in range(len(extended_lines)):
        for j in range(i+1, len(extended_lines)):
            try:
                pt1, pt2 = extended_lines[i]
                pt3, pt4 = extended_lines[j]
                m1, b1 = line_equation(pt1, pt2)
                m2, b2 = line_equation(pt3, pt4)
                intersection = line_intersection(m1, b1, m2, b2)
            except:
                print(f"Line 1: Point 1: {pt1}, Point 2: {pt2}, Slope: {round(m1, 2)}, Intercept: {round(b1, 2)}")
                print(f"Line 2: Point 3: {pt3}, Point 4: {pt4}, Slope: {round(m2, 2)}, Intercept: {round(b2, 2)}")
                intersection = None
            if intersection is not None:
                cv2.circle(image, intersection, 9, (0, 255, 0), -1)
                # print(intersection)

    cv2.imshow("Approx", image)
    # draw the card corners on the image
    if len(pts) == 4:
        cv2.circle(image, (int(pts[0][0][0]), int(pts[0][0][1])), 9, (0, 0, 255), -1)
        cv2.circle(image, (int(pts[1][0][0]), int(pts[1][0][1])), 9, (0, 255, 255), -1)
        cv2.circle(image, (int(pts[2][0][0]), int(pts[2][0][1])), 9, (255, 0, 255), -1)
        cv2.circle(image, (int(pts[3][0][0]), int(pts[3][0][1])), 9, (0, 255, 0), -1)

    

    # # Find width and height of card's bounding rectangle
    # x, y, w, h = cv2.boundingRect(contour)
    # card.width, card.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]
    w = 0
    h = 0
    # Warp card into 310x500 flattened image using perspective transform
    card.warp, image = flatten2(image, pts, w, h)

    # cv2.imshow("with corners", image)
    # cv2.imshow("Warp", card.warp)

    card.sign = get_sign(card)

    card.rank, card.suit = get_rank_and_suit(card)

    # Place the sign of the card on top of the debug view
    card.debug_view = card.warp.copy()
    cv2.putText(card.debug_view, card.sign, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(card.debug_view, card.rank, (10, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(card.debug_view, card.suit, (10, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return card

def find_longest_edges(approx, top_n=4):
    """
    Given an approxPolyDP contour, finds the top N longest edges.
    Returns a list of line segments [(pt1, pt2), ...] sorted by length.
    """
    edges = []
    for i in range(len(approx)):
        pt1 = tuple(approx[i][0])  # Current point
        pt2 = tuple(approx[(i + 1) % len(approx)][0])  # Next point (loop back to start)
        # Calculate Euclidean distance
        length = np.linalg.norm(np.array(pt1) - np.array(pt2))
        edges.append((length, pt1, pt2))
    
    # Sort by length (descending)
    edges.sort(reverse=True, key=lambda x: x[0])
    return [(edge[1], edge[2]) for edge in edges[:top_n]]  # Return only points,

def line_equation(pt1, pt2):
    """Returns (slope, intercept) for a line passing through pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 == x2:  # Vertical line case
        return None, x1  # None slope, x = constant
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def line_intersection(m1, b1, m2, b2):
    """Finds intersection of two lines given by y = mx + b."""
    if m1 == None:
        x = b1
        y = m2 * x + b2
        return int(x), int(y)
    if m2 == None:
        x = b2
        y = m1 * x + b1
        return int(x), int(y)
    if m1 == m2:  # Parallel lines (or both vertical)
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return int(x), int(y)

def extend_line(pt1, pt2, length=1000):
    """Extends a line segment far beyond its original length."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    scale = length / np.linalg.norm((dx, dy))
    new_pt1 = (int(x1 - dx * scale), int(y1 - dy * scale))
    new_pt2 = (int(x2 + dx * scale), int(y2 + dy * scale))
    # return line_equation(new_pt1, new_pt2)
    return new_pt1, new_pt2

def draw_line_from_equation(img, m, b, color=(0, 255, 0), thickness=2):
    height, width = img.shape[:2]
    # Compute y-values at x = 0 and x = width-1
    y1 = int(m * 0 + b)  # y when x = 0
    y2 = int(m * (width - 1) + b)  # y when x = width-1
    # Clip y-values to ensure they stay within image bounds
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    # Draw the line
    cv2.line(img, (0, y1), (width - 1, y2), color, thickness)

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
    # threshold again for some reason because the first time there were some little artifacts
    _, gray = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Get the number of distinct shapes
    actual_contours = []
    biggest_contour = 0
    for contour in contours:
        if cv2.contourArea(contour) > biggest_contour:
            biggest_contour = cv2.contourArea(contour)
        # check if area of contour is big enough to be distinct shape
        if cv2.contourArea(contour) > 1000:
            actual_contours.append(contour)

    # print(f"Biggest contour: {biggest_contour}")
    num_shapes = len(actual_contours)
    
    cv2.drawContours(image_with_contours, actual_contours, -1, (0, 255, 0), 2)

    # cv2.imshow("Contours", image_with_contours)

    # Determine suit
    shape = "Unknown"
    # Iterate through contours
    for contour in contours:
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(contour, True)

        """
        circle perimeter:       +- 160  (150-170)
        rectangle perimeter:    +- 180  (170-190)
        triangle perimeter:     +- 194  (190-200)
        """

        # Approximate the contour
        # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        approx = cv2.approxPolyDP(contour, 10, True)

        # Draw the approximated polygon
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(colour, [approx], True, (0, 255, 255), 2)  # Yellow for approximation
        # cv2.imshow(f"Debug View:", colour)

        # print(is_circle(contour))

        # Count the number of vertices
        vertices = len(approx)

        # Detect shape based on vertices
        if vertices == 3:
            shape = "Triangle"
            # print("Triangle perimeter:", peri)
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # print("Rectangle perimeter:", peri)
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle very long word and stuff to maybe notice what is happening" # in case of weird stuff
        else:
            # Detect circle by comparing area and perimeter
            # print("Circle perimeter:", peri)
            area = cv2.contourArea(contour)
            if area == 0:
                continue
            circularity = 4 * np.pi * (area / (peri * peri))
            if 0.8 <= circularity <= 1.2:  # Circularity close to 1
                shape = "Circle"
            else:
                shape = "Unknown"

    # if shape != "Square":
    #     print(f"Shape: {shape}")

    return str(num_shapes), shape

# def is_circle(contour):
#     """Determines if a contour is a circle based on circularity."""
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)
    
#     if perimeter == 0:
#         return False

#     circularity = 4 * np.pi * (area / (perimeter * perimeter))

#     value = 0.8 <= circularity <= 1.2  # Values close to 1 indicate a circle

#     if value == True:
#         print(circularity)

#     return value

def flatten(image, pts, w, h):
    """Flattens an image of a card into a top-down 310x500 perspective.
    Returns the flattened, re-sized image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    
    # choose top left point
    s = np.sum(pts, axis = 2)
    actual_tl_index = np.argmin(s)
    actual_tl = pts[actual_tl_index]

    # calculate distance to other points to check orientation
    distances = [0 for _ in range(len(pts))]
    for i, point in enumerate(pts):
        if i != actual_tl_index:
            distances[i] = calculate_distance(actual_tl, point)

    # find the point that is closest to the top left point
    actual_tr = pts[distances.index(min(distances))]

    # determine orientation of the points
    orientation = orientation(actual_tl, actual_tr)
    if orientation == "right":
        # card is upright

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

    elif orientation == "below":
        # card is rotated
        # top right smalles sum
        # bottom left largest sum
        s = np.sum(pts, axis = 2)
        tr = pts[np.argmin(s)]
        bl = pts[np.argmax(s)]

        # bottom right smallest diff
        # top left largest diff
        diff = np.diff(pts, axis = -1)
        br = pts[np.argmin(diff)]
        tl = pts[np.argmax(diff)]

    # make array of points in order of [top left, top right, bottom right, bottom left]
    temp_rect = np.array([tl, tr, br, bl], dtype="float32")

    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp

def flatten2(image, pts, w, h):
    """Flattens an image of a card into a top-down 310x500 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype = "float32")
    
    # top left smallest sum
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
        # print("vertical")
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        # print("horizontal")
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

    # draw the name of the corner on the image
    cv2.putText(image, f"top left", (int(temp_rect[0][0]), int(temp_rect[0][1])), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"bottom left", (int(temp_rect[1][0]), int(temp_rect[1][1])), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"bottom right", (int(temp_rect[2][0]), int(temp_rect[2][1])), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"top right", (int(temp_rect[3][0]), int(temp_rect[3][1])), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
            
    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp, image

def get_sign(card):
    """Determines the sign of the card based on overall color, 
    green for positive, red for negative"""

    # hue saturation value
    lower_hsv_red = np.array([0, 34, 46])
    upper_hsv_red = np.array([15, 109, 200])

    lower_hsv_green = np.array([32, 12, 12])
    upper_hsv_green = np.array([82, 162, 200])

    hsv = cv2.cvtColor(card.warp, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_hsv_red, upper_hsv_red)
    mask_green = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)

    # cv2.imshow("Red Mask", mask_red)
    # cv2.imshow("Green Mask", mask_green)

    pixels_red = cv2.countNonZero(mask_red)
    pixels_green = cv2.countNonZero(mask_green)

    return "Positive" if pixels_green > pixels_red else "Negative"


