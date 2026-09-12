import cv2, time, pyautogui
import mediapipe as mp

hand_module = mp.solutions.hands
hand_detector = hand_module.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75
)

drawing_tool = mp.solutions.drawing_utils

# Configurations
SCROLL_AMOUNT = 250
SCROLL_INTERVAL = 0.8
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480


def identify_gesture(hand_landmarks, hand_type):
    fingers_up = []

    finger_tips = [
        hand_module.HandLandmark.INDEX_FINGER_TIP,
        hand_module.HandLandmark.MIDDLE_FINGER_TIP,
        hand_module.HandLandmark.RING_FINGER_TIP,
        hand_module.HandLandmark.PINKY_TIP
    ]

    # Check fingers except thumb
    for finger_tip in finger_tips:
        if hand_landmarks.landmark[finger_tip].y < \
           hand_landmarks.landmark[finger_tip - 2].y:
            fingers_up.append(1)

    # Check thumb
    thumb_position = hand_landmarks.landmark[
        hand_module.HandLandmark.THUMB_TIP
    ]

    thumb_joint = hand_landmarks.landmark[
        hand_module.HandLandmark.THUMB_IP
    ]

    if (
        (hand_type == "Right" and thumb_position.x > thumb_joint.x)
        or
        (hand_type == "Left" and thumb_position.x < thumb_joint.x)
    ):
        fingers_up.append(1)

    # Open palm -> scroll up
    # Closed fist -> scroll down
    if sum(fingers_up) == 5:
        return "scroll_up"
    elif len(fingers_up) == 0:
        return "scroll_down"
    else:
        return "none"


# Start webcam
camera = cv2.VideoCapture(0)

camera.set(3, VIDEO_WIDTH)
camera.set(4, VIDEO_HEIGHT)

previous_scroll = 0
previous_time = 0

print(
    "Gesture Scroll System Active\n"
    "Open Palm: Scroll Up\n"
    "Closed Fist: Scroll Down\n"
    "Press 'e' to exit"
)


while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break

    # Convert BGR to RGB and flip horizontally
    frame = cv2.flip(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        1
    )

    # Process hand landmarks
    detection = hand_detector.process(frame)

    current_gesture = "none"
    current_hand = "Unknown"

    if detection.multi_hand_landmarks:

        for detected_hand, hand_info in zip(
            detection.multi_hand_landmarks,
            detection.multi_handedness
        ):

            current_hand = hand_info.classification[0].label

            current_gesture = identify_gesture(
                detected_hand,
                current_hand
            )

            # Draw hand landmarks
            drawing_tool.draw_landmarks(
                frame,
                detected_hand,
                hand_module.HAND_CONNECTIONS
            )


            # Perform scrolling after the delay
            if (time.time() - previous_scroll) > SCROLL_INTERVAL:

                if current_gesture == "scroll_up":
                    pyautogui.scroll(SCROLL_AMOUNT)

                elif current_gesture == "scroll_down":
                    pyautogui.scroll(-SCROLL_AMOUNT)

                previous_scroll = time.time()


    # Calculate FPS
    time_difference = time.time() - previous_time

    fps_value = (
        1 / time_difference
        if time_difference > 0
        else 0
    )

    previous_time = time.time()


    # Display information
    cv2.putText(
        frame,
        f"FPS: {int(fps_value)} | Hand: {current_hand} | Gesture: {current_gesture}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # Display webcam output
    cv2.imshow(
        "Hand Gesture Scrolling",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    )


    # Exit when 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


# Release resources
camera.release()
cv2.destroyAllWindows()
