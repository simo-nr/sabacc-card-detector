import cv2
import numpy as np

import time

start_time = time.time()


video_path = "media/test_vid_4.mov"
stream = cv2.VideoCapture(video_path)

# initialize previous edges
previous_edges = np.zeros((1, 1))
# prev_prev_edges = np.zeros((1, 1))

while True:

    ret, frame = stream.read()
    if not ret:
        break

    # Step 1: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Canny Edge Detection
    edges = cv2.Canny(gray, 110, 150)

    alpha = 0.5  # Adjust for smoother or sharper edges
    smoothed_edges = cv2.addWeighted(previous_edges, alpha, edges, 1 - alpha, 0)
    # prev_prev_edges = previous_edges.copy()  # Update for the next frame
    previous_edges = smoothed_edges.copy()  # Update for the next frame

    # cv2.imshow("Smoothed Edges", smoothed_edges)

    # Step 3: Apply Thresholding
    thresh_level = 175 # 190 for bright cards, 120 for dark cards, maybe 175 for bright cards because of smudges
    _, thresholded = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours in the edge-detected image
    contours, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Step 5: Create a mask that includes everything inside the detected contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # cv2.imshow("Mask", mask)

    # Step 6: Use the mask to filter the thresholded image
    filtered = cv2.bitwise_and(thresholded, thresholded, mask=mask)

    # Step 4: Combine both using bitwise OR
    # combined = cv2.bitwise_or(thresholded, edges)

    # Step 4: Use edges as a mask to filter the thresholded image
    # filtered = cv2.bitwise_and(thresholded, thresholded, mask=edges)


    # Display results
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Thresholded", thresholded)
    # cv2.imshow("Combined", combined)
    # cv2.imshow("Combined", filtered)
    cv2.imshow("contours", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))