import cv2
import numpy as np

import time

from flatten import flattener2

start_time = time.time()

TEMP_SMOOTHING = False
MORPH_OP = True
HOUGH_LINES = False
CONT_APPROX = False

video_path = "media/test_triangle.mov"
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

    if MORPH_OP:
        kernel = np.ones((3,3), np.uint8)  # Small kernel to avoid over-smoothing
        edges = cv2.Canny(gray, 110, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)  # Expand edges slightly
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close gaps

    if TEMP_SMOOTHING:
        alpha = 0.5  # Adjust for smoother or sharper edges
        edges = cv2.addWeighted(previous_edges, alpha, edges, 1 - alpha, 0)
        # prev_prev_edges = previous_edges.copy()  # Update for the next frame
        previous_edges = edges.copy()  # Update for the next frame

    if HOUGH_LINES:
        # Step 3: Apply Hough Line Transform to stabilize edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        # Create a blank image to draw lines on
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Get line coordinates
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw white lines
        # Combine edges and Hough lines
        edges = cv2.bitwise_or(edges, cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY))

    # cv2.imshow("Smoothed Edges", smoothed_edges)

    # Step 3: Apply Thresholding
    thresh_level = 175 # 190 for bright cards, 120 for dark cards, maybe 175 for bright cards because of smudges
    _, thresholded = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if CONT_APPROX:
        # Step 6: Approximate contours to reduce unnecessary detail
        approx_contours = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)  # Adjust for more/less simplification
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)
    else:
        approx_contours = contours

    # draw contours
    # cv2.drawContours(frame, approx_contours, -1, (0, 255, 0), 2)

    # Step 5: Create a mask that includes everything inside the detected contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, approx_contours, -1, (255), thickness=cv2.FILLED)

    # cv2.imshow("Mask", mask)

    # Step 6: Use the mask to filter the thresholded image
    filtered = cv2.bitwise_and(thresholded, thresholded, mask=mask)

    # Step 4: Combine both using bitwise OR
    # combined = cv2.bitwise_or(thresholded, edges)

    # Step 4: Use edges as a mask to filter the thresholded image
    # filtered = cv2.bitwise_and(thresholded, thresholded, mask=edges)

    # detect contours in filtered image
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw contours on original frame
    cv2.drawContours(frame, contours, -1, (0, 100, 255), 2)

    # Display results
    cv2.imshow("Edges", edges)
    cv2.imshow("Thresholded", thresholded)
    # cv2.imshow("Combined", combined)
    cv2.imshow("Combined", filtered)
    cv2.imshow("contours", frame)

    # flatten detected contours and display
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.01*peri,True)
        pts = np.float32(approx)

        flattened, frame = flattener2(frame, pts, w, h)
        cv2.imshow(f"Flattened: {i}", flattened)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))