import cv2
import numpy as np
import VideoStream
from Cards import Card
import Cards


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 200000
CARD_MIN_AREA = 5000

font = cv2.FONT_HERSHEY_SIMPLEX

video_path = "test_vid.mov"
videostream = VideoStream.VideoStream(video_path).start()


def main():
    cam_quit = 0 # Loop control variable
    while cam_quit == 0:
        # Grab frame from video stream
        frame = videostream.read()

        if frame is None:
            print("End of video")
            break
        
        # Preprocess the frame (gray, blur, and threshold it)
        pre_proc = preprocess_frame(frame)
        # Find and sort contours of card in the frame
        cnts_sort, cnt_is_card = find_cards(pre_proc)

        # Draw card contours on image if contour is card
        # (have to do contours all at once or they do not show up properly for some reason)
        cards = []
        k = 0
        if len(cnts_sort) != 0:

            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    cards.append(preprocess_card(cnts_sort[i], frame))
                    frame = draw_results(frame, cards[k])
                    k = k + 1
            
            if len(cards) != 0:
                tmp_cnts = []
                for i, card in enumerate(cards):
                    # cv2.imshow(f"Card: {i}", card.debug_view)
                    tmp_cnts.append(card.contour)
                cv2.drawContours(frame, tmp_cnts, -1, (255,0,0), 2)

        # Show the video feed
        # cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam_quit = 1

    cv2.destroyAllWindows()
    videostream.stop()

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_level = 190 # 190 for bright cards, 120 for dark cards
    _, thresh = cv2.threshold(blurred, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh

def find_cards(frame):
    # Find contours in the tresholded image
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If there are no contours, do nothing
    if len(contours) == 0:
        return [], []
    
    # sort contour indices by contour size
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]), reverse=True)

    # Initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(contours),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy.
    # Now, the indices of the contour list still correspond with those of the hierarchy list. 
    # The hierarchy array can be used to check if the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(contours[i])
        hier_sort.append(hierarchy[0][i])

    # Determine which of the contours are cards by applying the following criteria: 
    # 1) Smaller area than the maximum card size
    # 2) Bigger area than the minimum card size
    # 3) Have no parents
    # 4) Have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

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

    card.sign = Cards.get_sign(card)

    # Place the sign of the card on top of the debug view
    card.debug_view = card.warp.copy()
    cv2.putText(card.debug_view, card.sign, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
    if num_shapes != 2 and num_shapes != 3:
        print(f"Number of distinct shapes: {num_shapes}")
        cv2.imwrite(f"wrong_rank_{num_shapes}.png", image_with_contours)


    cv2.imshow("Contours", image_with_contours)
    # cv2.imshow("Rank", card.rank_img)

    return card

def flatten(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
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

def draw_results(frame, card):
    """Draw the card name, center point, and contour on the camera image."""

    x = card.center[0]
    y = card.center[1]
    cv2.circle(frame, (x,y), 5, (255,0,0), -1)

    rank_name = "3"
    suit_name = "squares"
    card_sign = card.sign

    # Draw card name twice, so letters have black outline
    cv2.putText(frame, (rank_name+' of'), (x-60,y-10), font, 1, (50,200,200), 2, cv2.LINE_AA)
    cv2.putText(frame, (rank_name+' of'), (x-60,y-10), font, 1, (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(frame, card_sign, (x-60,y-25), font, 1, (50,200,200), 2, cv2.LINE_AA)

    cv2.putText(frame, suit_name, (x-60,y+25), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, suit_name, (x-60,y+25), font, 1, (50,200,200), 2, cv2.LINE_AA)

    return frame

if __name__ == "__main__":
    main()
