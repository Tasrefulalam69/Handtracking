import cv2
import numpy as np

# Open Webcam
cap = cv2.VideoCapture(0)

print("Hand Tracking & Gesture Recognition Started... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural view
    frame = cv2.flip(frame, 1)
    
    # Define ROI (Region of Interest) for hand placement
    cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
    roi = frame[100:400, 300:600]

    # Convert ROI to HSV color space for Skin Segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define Range for Skin Color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Blur the mask to remove noise
    blur = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in ROI
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find maximum contour (the hand)
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))

        if cv2.contourArea(max_contour) > 1000:
            # Draw contour inside ROI
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Convex Hull and Landmarks
            hull = cv2.convexHull(max_contour)
            cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

            cv2.putText(frame, "Hand Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Place Hand in Green Box", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display Windows
    cv2.imshow("Project 1 - Hand Gesture Tracking", frame)
    cv2.imshow("Skin Mask", blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()