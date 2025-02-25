import cv2
import numpy as np
import VideoStream
import Cards
import time


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 200000
CARD_MIN_AREA = 5000

CARD_HISTORY = 5

font = cv2.FONT_HERSHEY_SIMPLEX

global video_path
video_path = "media/test_triangle.mov"
global videostream
videostream = VideoStream.VideoStream(video_path).start()


def main():
    detected_cards = [set() for _ in range(CARD_HISTORY)] # Keep track of cards detected in previous frames
    last_message_send = []
    frame_counter = 0

    cam_quit = 0 # Loop control variable
    while cam_quit == 0:
        # Grab frame from video stream
        frame = videostream.read()

        if frame is None:
            print("End of video")
            break
        
        # Preprocess the frame (gray, blur, and threshold it)
        pre_proc = preprocess_frame(frame)
        cv2.imshow("Preprocessed", pre_proc)

        # Find and sort contours of card in the frame
        cnts_sort, cnt_is_card = find_cards(pre_proc)

        # Draw contours found by find_cards on the preprocessed frame and show it
        # full_contours = cv2.drawContours(pre_proc.copy(), [cnts_sort[i] for i in range(len(cnts_sort)) if cnt_is_card[i] == 1], -1, (255,255,0), 2)
        # cv2.imshow("All contours", full_contours)

        # Draw card contours on image if contour is card
        # (have to do contours all at once or they do not show up properly for some reason)
        cards = []
        k = 0
        if len(cnts_sort) != 0:

            detected_cards.append(set())

            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    cards.append(Cards.preprocess_card(cnts_sort[i], frame))
                    frame = draw_results(frame, cards[k])

                    # append cards in last frame with info of card
                    detected_cards[-1].add((cards[k].sign, cards[k].rank, cards[k].suit))

                    k = k + 1
            
            # draw all the card contours together
            if len(cards) != 0:
                tmp_cnts = []
                for i, card in enumerate(cards):
                    cv2.imshow(f"Card: {i}", card.debug_view)
                    # cv2.imshow(f"Card: {i}", card.warp)
                    tmp_cnts.append(card.contour)
                cv2.drawContours(frame, tmp_cnts, -1, (255,0,0), 2)
        
        # check list of previous frames
        if len(detected_cards) > CARD_HISTORY:
            detected_cards.pop(0)

        """
        # check if all items in detected_cards are the same
        if all(s == detected_cards[0] for s in detected_cards) and detected_cards[0] != set():
            # print(f"Cards are the same in the last {len(detected_cards)} frames: frame {frame_counter}")
            if last_message_send != detected_cards[-1]:
                last_message_send = detected_cards[-1]
                print("send message: ", last_message_send)
        """

        # debug stuff
        # if detected_cards[0] != {('Positive', '4', 'Triangle')}:
        #     print(detected_cards[0])
        
        # Show the video feed
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam_quit = 1
        
        frame_counter += 1
        # time.sleep(0.25)
        
    cv2.destroyAllWindows()
    videostream.stop()

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_level = 175 # 190 for bright cards, 120 for dark cards, maybe 175 for bright cards because of smudges
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
    cnt_is_card = np.zeros(len(contours), dtype=int)

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

def draw_results(frame, card):
    """Draw the card name, center point, and contour on the camera image."""

    x = card.center[0]
    y = card.center[1]
    cv2.circle(frame, (x,y), 5, (255,0,0), -1)

    rank_name = card.rank
    suit_name = card.suit
    card_sign = card.sign

    # Draw card name twice, so letters have black outline
    cv2.putText(frame, card_sign, (x-60,y-45), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, card_sign, (x-60,y-45), font, 1, (50,200,200), 2, cv2.LINE_AA)

    cv2.putText(frame, (rank_name+' of'), (x-60,y-10), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, (rank_name+' of'), (x-60,y-10), font, 1, (50,200,200), 2, cv2.LINE_AA)

    cv2.putText(frame, suit_name, (x-60,y+25), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, suit_name, (x-60,y+25), font, 1, (50,200,200), 2, cv2.LINE_AA)

    return frame

if __name__ == "__main__":
    # for i in range(4):
    #     video_path = f"media/test_vid_{i+1}.mov"
    #     videostream = VideoStream.VideoStream(video_path).start()
    main()
