import numpy as np
import cv2
import math
from typing import List, Tuple, Optional

import time


font = cv2.FONT_HERSHEY_SIMPLEX

class Card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 310x500, flattened, grayed, blurred image
        self.sign = "Unknown" # positive or negative
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.rank = "Unknown" # Rank of card
        self.suit = "Unknown" # Suit of card
        
        self.debug_view = [] # Debug view of card


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

    # find 4 longest lines in approx
    longest_edges = find_longest_edges(approx, 4)

    # extend the lines
    extended_lines = []
    for pt1, pt2 in longest_edges:
        p1, p2 = extend_line(pt1, pt2)
        extended_lines.append((p1, p2))
        # draw_line_from_equation(image, m, b)
        # cv2.line(image, p1, p2, (255, 0, 0), 1)

    # find the intersection of the lines
    intersections = []
    for i in range(len(extended_lines)):
        for j in range(i+1, len(extended_lines)):
            try:
                pt1, pt2 = extended_lines[i]
                pt3, pt4 = extended_lines[j]
                m1, b1 = line_equation(pt1, pt2)
                m2, b2 = line_equation(pt3, pt4)
                intersections.append([line_intersection(m1, b1, m2, b2)])
            except:
                print(f"Line 1: Point 1: {pt1}, Point 2: {pt2}, Slope: {round(m1, 2)}, Intercept: {round(b1, 2)}")
                print(f"Line 2: Point 3: {pt3}, Point 4: {pt4}, Slope: {round(m2, 2)}, Intercept: {round(b2, 2)}")

    intersections = [pt for pt in intersections if pt[0] is not None and 
                 0 <= pt[0][0] <= 1920 and 0 <= pt[0][1] <= 1080]
    
    # TODO better intersection filter, if still longer than 4, remove furthest outlier

    pts = np.float32(intersections)
    card.corner_pts = pts
    if len(pts) != 4:
        print(f"\033[93mIntersections: \n{pts}\033[0m")
        card.debug_view = None
        return card

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]

    mod_image = image.copy()
    # Warp card into 310x500 flattened image using perspective transform
    card.warp, mod_image = flatten(mod_image, pts)

    card.sign = get_sign(card)
    card.rank, card.suit = get_rank_and_suit(card)

    ##################### DEBUG #####################
    p1 = (int(pts[0][0][0]), int(pts[0][0][1]))
    p2 = (int(pts[1][0][0]), int(pts[1][0][1]))
    p3 = (int(pts[2][0][0]), int(pts[2][0][1]))
    p4 = (int(pts[3][0][0]), int(pts[3][0][1]))

    sum_loc_p1 = (int(pts[0][0][0]), int(pts[0][0][1]) - 20)
    sum_loc_p2 = (int(pts[1][0][0]), int(pts[1][0][1]) - 20)
    sum_loc_p3 = (int(pts[2][0][0]), int(pts[2][0][1]) - 20)
    sum_loc_p4 = (int(pts[3][0][0]), int(pts[3][0][1]) - 20)

    cv2.circle(mod_image, p1, 9, (0, 0, 255), -1)
    cv2.circle(mod_image, p2, 9, (0, 255, 255), -1)
    cv2.circle(mod_image, p3, 9, (255, 0, 255), -1)
    cv2.circle(mod_image, p4, 9, (0, 255, 0), -1)

    sum_p1 = np.sum(p1)
    sum_p2 = np.sum(p2)
    sum_p3 = np.sum(p3)
    sum_p4 = np.sum(p4)

    cv2.putText(mod_image, f"{p1}", p1, font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{p2}", p2, font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{p3}", p3, font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{p4}", p4, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(mod_image, f"{sum_p1}", sum_loc_p1, font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{sum_p2}", sum_loc_p2, font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{sum_p3}", sum_loc_p3, font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(mod_image, f"{sum_p4}", sum_loc_p4, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imshow("mod by flat", mod_image)

    # Place the sign of the card on top of the debug view
    card.debug_view = card.warp.copy()
    cv2.putText(card.debug_view, card.sign, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(card.debug_view, card.rank, (10, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(card.debug_view, card.suit, (10, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    ################## END OF DEBUG ##################

    # if card.rank != "4" or card.suit != "Triangle":
    #     cv2.imwrite(f"junk_stuff/{time.time()}_mod_image.png", mod_image)
    #     cv2.imwrite(f"junk_stuff/{time.time()}_warp.png", card.warp)

    return card

def get_rank_and_suit(card: Card) -> Tuple[str, str]:
    """
    Determines rank and suit of card based on warp view of card. 
    Looks for the number of contours and shape of biggest contour on the center of the card.
    """
    # Define the coordinates for the region of interest (ROI)
    x_start, y_start = 55, 115  # Starting coordinates (top-left corner)
    x_end, y_end = 255, 400     # Ending coordinates (bottom-right corner)

    # # place dots on the image to show the ROI
    # cv2.circle(card.warp, (x_start, y_start), 9, (0, 0, 255), -1)
    # cv2.circle(card.warp, (x_end, y_end), 9, (0, 0, 255), -1)
    # cv2.imshow("ROI", card.warp)
    
    # Crop and treshold the rank image from the card
    rank_img = card.warp[y_start:y_end, x_start:x_end]

    thresh_level: int = 200
    _, card.rank_img = cv2.threshold(rank_img, thresh_level, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(card.rank_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # need to threshold again for some reason because the previous thresholding looks weird
    _, gray = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # # blur
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Get the number of distinct shapes
    actual_contours: List = []
    biggest_area: int = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = cv2.contourArea(contour)
            biggest_contour = contour
        # check if area of contour is big enough to be distinct shape
        if area > 1000:
            actual_contours.append(contour)

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
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        # approx = cv2.approxPolyDP(contour, 10, True)

        # Draw the approximated polygon
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # cv2.polylines(colour, [approx], True, (0, 255, 255), 2)  # Yellow for approximation
        # cv2.imshow(f"Debug View:", colour)

        # Count the number of vertices
        vertices = len(approx)

        # # Detect shape based on vertices
        # if vertices == 3:
        #     shape = "Triangle"
        # elif vertices == 4:
        #     approx = cv2.approxPolyDP(biggest_contour, 0.03 * peri, True)
        #     longest_edge = find_longest_edges(approx, 1)[0]
        #     p1, p2 = longest_edge
        #     length = calculate_distance(p1, p2)
        #     # print("length: ", length)

        #     # cv2.polylines(colour, [approx], True, (0, 255, 255), 2)  # Yellow for approximation
        #     # # draw longest edge
        #     # cv2.line(colour, p1, p2, (0, 0, 255), 2)
        #     # cv2.imshow(f"Debug View:", colour)

        #     if length >= 25: # might be cutting it close, would prefer 30 (currently 0.03 * peri approx)
        #         shape = "Square"
        #     else:
        #         if card.sign == "Negative":
        #             print("length: ", length)
        #         cv2.imwrite(f"junk_stuff/{length}_{int(time.time())}_approx.png", colour)
        #         shape = "Circle"
        # else:
        #     # Detect circle by comparing area and perimeter
        #     area = cv2.contourArea(contour)
        #     if area == 0:
        #         continue
        #     circularity = 4 * np.pi * (area / (peri * peri))
        #     if 0.8 <= circularity <= 1.2:  # Circularity close to 1
        #         if card.sign == "Negative":
        #             print("circularity: ", circularity)
        #         shape = "Circle"
        #     else:
        #         shape = "Unknown"

        # Detect shape based on vertices
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            # Further check if it's square or rectangle
            # x, y, w, h = cv2.boundingRect(approx)
            # aspect_ratio = float(w) / h
            # if 0.9 <= aspect_ratio <= 1.1:  # Almost square
            #     shape = "Square"
            # else:
            #     shape = "Rectangle"  # Not needed for your specific case
            shape = "Square"
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

def get_sign(card: Card) -> str:
    """Determines the sign of the card based on overall color, 
    green for positive, red for negative"""

    # hue saturation value
    lower_hsv_red = np.array([0, 34, 46])
    upper_hsv_red = np.array([15, 170, 200])

    lower_hsv_green = np.array([32, 12, 12])
    upper_hsv_green = np.array([82, 162, 200])

    hsv = cv2.cvtColor(card.warp, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_hsv_red, upper_hsv_red)
    mask_green = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)


    pixels_red = cv2.countNonZero(mask_red)
    pixels_green = cv2.countNonZero(mask_green)

    # if pixels_red > pixels_green:
    #     cv2.imshow("Red Mask", mask_red)
    #     cv2.imshow("Green Mask", mask_green)

    return "Positive" if pixels_green > pixels_red else "Negative"

def flatten(image, pts):
    """
    Flattens an image of a card into a top-down 310x500 perspective.
    Returns the flattened, re-sized image.
    """

    pts = pts.reshape(4, 2)
    
    # choose top left point, smallest sum of coordinates
    s = np.sum(pts, axis = 1)
    tl_index = np.argmin(s)
    tl = pts[tl_index]

    # calculate distance to other points to check orientation
    distances = [0 for _ in range(len(pts))]
    for i, point in enumerate(pts):
        if i != tl_index:
            distances[i] = calculate_distance(tl, point)

    # find the point that is closest to the top left point
    # as long as point along the short length is closer, the card will be stretched the right way
    closest = pts[distances.index(min([dist for dist in distances if dist > 0]))]
    furthest = pts[distances.index(max(distances))]
    tr = closest
    br = furthest

    # remove tl, tr and br from pts
    pts = pts[~np.all(pts == tl, axis=1)]
    pts = pts[~np.all(pts == tr, axis=1)]
    pts = pts[~np.all(pts == br, axis=1)]

    # bottom left is remaining point
    if len(pts) != 1:
        print("points: ", pts)
    bl = pts[0]
    
    # make array of points in order of [top left, top right, bottom right, bottom left]
    temp_rect = np.zeros((4, 2), dtype = "float32")
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl

    # draw points from temp_rect on image
    for point in temp_rect:
        x, y = point
        cv2.circle(image, (int(x), int(y)), 9, (0, 0, 255), -1)

    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # draw the name of the corner on the image
    cv2.putText(image, f"top left", (int(temp_rect[0][0]), int(temp_rect[0][1]) + 20), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"top right", (int(temp_rect[1][0]), int(temp_rect[1][1]) + 20), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"bottom right", (int(temp_rect[2][0]), int(temp_rect[2][1]) + 20), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"bottom left", (int(temp_rect[3][0]), int(temp_rect[3][1]) + 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return warp, image

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def find_longest_edges(approx: np.ndarray, top_n: int = 4) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Given an approxPolyDP contour, finds the top N longest edges.
    Returns a list of line segments [(pt1, pt2), ...] sorted by length.
    """
    edges: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
    for i in range(len(approx)):
        pt1: Tuple[int, int] = tuple(approx[i][0])  # Current point
        pt2: Tuple[int, int] = tuple(approx[(i + 1) % len(approx)][0])  # Next point (loop back to start)
        # Calculate Euclidean distance
        length: float = np.linalg.norm(np.array(pt1) - np.array(pt2))
        edges.append((length, pt1, pt2))
    
    # Sort by length (descending)
    edges.sort(reverse=True, key=lambda x: x[0])
    return [(edge[1], edge[2]) for edge in edges[:top_n]]  # Return only points

def line_equation(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> Tuple[Optional[float], float]:
    """Returns (slope, intercept) for a line passing through pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 == x2:  # Vertical line case
        return None, float(x1)  # None slope, x = constant
    m: float = (y2 - y1) / (x2 - x1)
    b: float = y1 - m * x1
    return m, b

def line_intersection(m1: Optional[float], b1: float, m2: Optional[float], b2: float) -> Optional[Tuple[int, int]]:
    """Finds intersection of two lines given by y = mx + b."""
    if m1 is None and m2 is None:  # Both lines are vertical
        return None
    if m1 is None:
        x: float = b1
        y: float = m2 * x + b2  # type: ignore
        return int(x), int(y)
    if m2 is None:
        x: float = b2
        y: float = m1 * x + b1  # type: ignore
        return int(x), int(y)
    if m1 == m2:  # Parallel lines (or both vertical)
        return None
    x: float = (b2 - b1) / (m1 - m2)
    y: float = m1 * x + b1
    return int(x), int(y)

def extend_line(pt1: Tuple[int, int], pt2: Tuple[int, int], length: int = 1000) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Extends a line segment far beyond its original length."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    scale: float = length / np.linalg.norm((dx, dy))
    new_pt1: Tuple[int, int] = (int(x1 - dx * scale), int(y1 - dy * scale))
    new_pt2: Tuple[int, int] = (int(x2 + dx * scale), int(y2 + dy * scale))
    return new_pt1, new_pt2

