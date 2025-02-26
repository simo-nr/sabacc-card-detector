import cv2
import numpy as np
import math
import time

start_time = time.time()

# Load the image
# image = cv2.imread("media/test_img_11.png")
video_path = "media/test_triangle.mov"
stream = cv2.VideoCapture(video_path)

def flattener2(image, pts, w, h):
    """Flattens an image of a card into a top-down 310x500 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")

    print(f"Points: {pts}")

    # print(f"Points: {pts}")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        # print("Vertical")
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        # print("Horizontal")
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
            # print("Diamond, tilted left")
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # print("Diamond, tilted right")
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left

    # draw on image based on temp_rect coordinates
    # write top left
    image = cv2.circle(image, (int(temp_rect[0][0]), int(temp_rect[0][1])), 9, (0, 0, 255), -1)
    image = cv2.putText(image, "Top Left", (int(temp_rect[0][0]), int(temp_rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # write top right
    image = cv2.circle(image, (int(temp_rect[1][0]), int(temp_rect[1][1])), 9, (0, 255, 255), -1)
    image = cv2.putText(image, "Top Right", (int(temp_rect[1][0]), int(temp_rect[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    # write bottom right
    image = cv2.circle(image, (int(temp_rect[2][0]), int(temp_rect[2][1])), 9, (0, 255, 0), -1)
    image = cv2.putText(image, "Bottom Right", (int(temp_rect[2][0]), int(temp_rect[2][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # write bottom left
    image = cv2.circle(image, (int(temp_rect[3][0]), int(temp_rect[3][1])), 9, (255, 0, 255), -1)
    image = cv2.putText(image, "Bottom Left", (int(temp_rect[3][0]), int(temp_rect[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    
    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp, image

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def get_orientation_relative(p1, p2):
    """Returns orientation of p1 relative to p2."""
    dx = abs(p2[0] - p1[0])  # Horizontal distance
    dy = abs(p2[1] - p1[1])  # Vertical distance

    if dy > dx:
        if p2[1] < p1[1]:
            return "above"
        else:
            return "below"
    else:
        if p2[0] < p1[0]:
            return "left"
        else:
            return "right"


def flattener(image, pts, w, h):
    # choose top left point
    s = np.sum(pts, axis = 2)
    actual_tl_index = np.argmin(s)
    actual_tl = pts[actual_tl_index]

    image = cv2.circle(image, actual_tl[0], radius=9, color=(0, 0, 255), thickness=-1)

    # calculate distance to other points to check orientation
    distances = [0 for _ in range(len(pts))]
    for i, point in enumerate(pts):
        if i != actual_tl_index:
            distances[i] = calculate_distance(actual_tl[0], point[0])

    # find the point that is closest to the top left point
    actual_tr = pts[distances.index(min([d for d in distances if d > 0]))]

    image = cv2.circle(image, actual_tr[0], radius=9, color=(255, 0, 0), thickness=-1)

    distances.pop(distances.index(max(distances)))
    # print()

    # determine orientation of the points
    card_orientation = get_orientation_relative(actual_tl[0], actual_tr[0])

    if card_orientation == "right":
        # card is upright
        # print("Upright")

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

    elif card_orientation == "below":
        # card is rotated
        # print("Rotated")
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

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform
    temp_rect = np.zeros((4,2), dtype = "float32")

    temp_rect[0] = tl[0]
    temp_rect[1] = tr[0]
    temp_rect[2] = br[0]
    temp_rect[3] = bl[0]

    maxWidth = 310
    maxHeight = 500

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp, image


def get_orientation(w, h):
    """Determine the orientation of the card based on its bounding rectangle."""

    # w <= 0.8*h: vertically oriented
    # w >= 1.2*h: horizontally oriented
    # 0.8*h < w < 1.2*h: diamond oriented

    if w >= 1.2 * h:
        return "horizontal"
    elif w <= 0.8 * h:
        return "vertical"
    else:
        return "diagonal"

if __name__ == "__main__":
    # Example usage in your code
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        # Convert the frame to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        # edged = cv2.Canny(blur, 75, 200)
        thresh_level = 175 # 190 for bright cards, 120 for dark cards, maybe 175 for bright cards because of smudges
        _, threshold = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

        cv2.imshow("Threshold", threshold)

        # Find contours in the edged image
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a variable to store the largest contour
        largest_contour = None
        max_area = 0

        # Loop over the contours
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # If the contour has four points, it is likely to be a rectangle
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    largest_contour = approx
                    print(f"Approx: {approx}")
                    max_area = area

        # If a largest contour was found, flatten it and show the result
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            card_orientation = get_orientation(w, h)
            # print(f"Card orientation: {card_orientation}")

            flattened2, frame = flattener2(frame, largest_contour, w, h)
            cv2.imshow("Flattened2", flattened2)

            # flattened, frame = flattener(frame, largest_contour, w, h)
            # cv2.imshow("Flattened", flattened)

        # Show the original frame with contours
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    stream.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")