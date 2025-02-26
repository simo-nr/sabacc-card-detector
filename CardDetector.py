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

    start_time = time.time()

    fetch_frame_times = []
    preprocess_time = []
    frame_times = []
    detection_times = []
    contour_times = []
    for_1_times = []
    for_2_times = []

    prev_edges = np.zeros((1, 1))
    cam_quit = 0 # Loop control variable
    while cam_quit == 0:
        frame_start_time = time.time()
        # Grab frame from video stream
        fetch_start_time = time.time()
        frame = videostream.read()
        fetch_frame_times.append(time.time() - fetch_start_time)
        if frame is None:
            print("End of video")
            break
        
        # Preprocess the frame (gray, blur, and threshold it)
        start_preprocess_time = time.time()
        pre_proc, prev_edges = preprocess_frame(frame, prev_edges)
        end_preprocess_time = time.time()
        preprocess_time.append(end_preprocess_time - start_preprocess_time)
        # print(f"Preprocessing time: {preprocess_time:.4f} seconds")
        cv2.imshow("Preprocessed", pre_proc)

        detection_start_time = time.time()
        # Find and sort contours of card in the frame
        cnts_sort, cnt_is_card = find_cards(pre_proc)
        detection_end_time = time.time()
        detection_times.append(detection_end_time - detection_start_time)

        contour_start_time = time.time()
        # Draw contours found by find_cards on the preprocessed frame and show it
        full_contours = cv2.drawContours(frame.copy(), [cnts_sort[i] for i in range(len(cnts_sort)) if cnt_is_card[i] == 1], -1, (255,255,0), 2)
        cv2.imshow("All contours", full_contours)
        contour_times.append(time.time() - contour_start_time)

        # Draw card contours on image if contour is card
        # (have to do contours all at once or they do not show up properly for some reason)
        cards = []
        k = 0
        if len(cnts_sort) != 0:

            detected_cards.append(set())

            for_1_time = time.time()

            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    cards.append(Cards.preprocess_card(cnts_sort[i], frame))
                    frame = draw_results(frame, cards[k])

                    # MESSAGE TO NN STUFF: append cards in last frame with info of card
                    detected_cards[-1].add((cards[k].sign, cards[k].rank, cards[k].suit))
                    k = k + 1

            for_1_times.append(time.time() - for_1_time)
            
            for_2_time = time.time()
            # draw all the card contours together
            if len(cards) != 0:
                tmp_cnts = []
                for i, card in enumerate(cards):
                    cv2.imshow(f"Card: {i}", card.debug_view)
                    # cv2.imshow(f"Card: {i}", card.warp)
                    tmp_cnts.append(card.contour)
                cv2.drawContours(frame, tmp_cnts, -1, (255,0,0), 2)
            
            for_2_times.append(time.time() - for_1_time)
        
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
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)

        # time.sleep(0.25)

    total_time_taken = time.time() - start_time
    average_frame_time = sum(frame_times) / len(frame_times)
    average_preprocess_time = sum(preprocess_time) / len(preprocess_time)
    average_detection_time = sum(detection_times) / len(detection_times)
    total_preprocess_time = sum(preprocess_time)

    percent_detection = average_detection_time / average_frame_time
    percent_preprocess = average_preprocess_time / average_frame_time

    average_for_1_time = sum(for_1_times) / len(for_1_times)
    average_for_2_time = sum(for_2_times) / len(for_2_times)

    percent_for_1 = average_for_1_time / average_frame_time
    percent_for_2 = average_for_2_time / average_frame_time

    average_contour_time = sum(contour_times) / len(contour_times)
    percent_contour_time = average_contour_time / average_frame_time

    print(f"\033[91mtime taken: {total_time_taken}\033[0m")
    print(f"\033[91maverage frame time: {average_frame_time} seconds\033[0m")
    print(f"\033[91maverage preprocess time: {average_preprocess_time} seconds\033[0m")
    print(f"\033[91maverage card detection time: {average_detection_time} seconds\033[0m")
    print(f"\033[91mtotal frames: {frame_counter}\033[0m")
    print(f"\033[91mtotal processing time: {total_preprocess_time} seconds\033[0m")

    print(f"\033[92mpercent preprocess time: {percent_preprocess}\033[0m")
    print(f"\033[92mpercent detection time: {percent_detection}\033[0m")

    print(f"\033[93maverage for_1 time: {average_for_1_time}\033[0m")
    print(f"\033[93maverage for_2 time: {average_for_2_time}\033[0m")

    print(f"contour start time: {average_contour_time}")

    print(f"\033[92mpercent for 1: {percent_for_1}\033[0m")
    print(f"\033[92mpercent for 2: {percent_for_2}\033[0m")

    print(f"percent contour time: {percent_contour_time}")

    total_percent = percent_preprocess + percent_detection + percent_for_1 + percent_for_2 + percent_contour_time
    print(f"\033[92mtotal percent: {total_percent}\033[0m")

    average_fetch_time = sum(fetch_frame_times) / len(fetch_frame_times)
    print(f"average fetch frame time: {average_fetch_time}")
    print(f"percent fetch time: {average_fetch_time / average_frame_time}")
        
    cv2.destroyAllWindows()
    videostream.stop()

def preprocess_frame(frame, previous_edges=None):
    """Preprocess the frame by applying Gaussian blur, Canny edge detection, and thresholding."""
    # Step 1: Apply Gaussian Blur and convert to grayscale
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Canny Edge Detection
    edges = cv2.Canny(gray, 110, 150)

    MORPH_OP = True
    TEMP_SMOOTHING = True

    if MORPH_OP:
        kernel = np.ones((3,3), np.uint8)  # Small kernel to avoid over-smoothing
        edges = cv2.Canny(gray, 110, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)  # Expand edges slightly
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close gaps

    if TEMP_SMOOTHING:
        if previous_edges is None:
            previous_edges = np.zeros((1, 1))
        alpha = 0.5  # Adjust for smoother or sharper edges
        edges = cv2.addWeighted(previous_edges, alpha, edges, 1 - alpha, 0)
        # prev_prev_edges = previous_edges.copy()  # Update for the next frame
        previous_edges = edges.copy()  # Update for the next frame
    
    # Step 3: Apply Thresholding
    thresh_level = 175 # 190 for bright cards, 120 for dark cards, maybe 175 for bright cards because of smudges
    _, thresholded = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create a mask that includes everything inside the detected contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Step 6: Use the mask to filter the thresholded image
    filtered = cv2.bitwise_and(thresholded, thresholded, mask=mask)

    return filtered, edges

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
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    # TODO: remove frame from return
    return cnts_sort, cnt_is_card#, frame

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
