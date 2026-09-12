import cv2
import numpy as np

# Set up webcam capture
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access webcam.")
    exit()

while True:
    # Capture frame-by-frame
    success, image = camera.read()

    if not success:
        print("Error: Unable to capture frame.")
        break

    # Convert to HSV for color filtering
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for skin color in HSV
    skin_lower = np.array([0, 25, 60], dtype=np.uint8)
    skin_upper = np.array([25, 255, 255], dtype=np.uint8)

    # Create a mask to detect skin color
    skin_mask = cv2.inRange(hsv_frame, skin_lower, skin_upper)

    # Apply the mask to the frame
    filtered_result = cv2.bitwise_and(
        image,
        image,
        mask=skin_mask
    )

    # Find contours of the detected hand
    hand_contours, _ = cv2.findContours(
        skin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # If contours are found, process the largest one
    if hand_contours:
        largest_hand = max(
            hand_contours,
            key=cv2.contourArea
        )

        # Ignore very small contours
        if cv2.contourArea(largest_hand) > 600:

            # Draw the bounding box around the detected hand
            pos_x, pos_y, box_width, box_height = cv2.boundingRect(
                largest_hand
            )

            cv2.rectangle(
                image,
                (pos_x, pos_y),
                (pos_x + box_width, pos_y + box_height),
                (255, 0, 0),
                2
            )

            # Get the center of the detected hand
            hand_center_x = int(pos_x + box_width / 2)
            hand_center_y = int(pos_y + box_height / 2)

            cv2.circle(
                image,
                (hand_center_x, hand_center_y),
                6,
                (0, 0, 255),
                -1
            )  # Red dot at center

    # Display the original and filtered frames
    cv2.imshow('Camera View', image)
    cv2.imshow('Skin Filter Result', filtered_result)

    # Break the loop when 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

camera.release()
cv2.destroyAllWindows()
