import cv2, mediapipe as mp, numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

HandModule = mp.solutions.hands
hand_detector = HandModule.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

drawing = mp.solutions.drawing_utils

THUMB = HandModule.HandLandmark.THUMB_TIP
INDEX = HandModule.HandLandmark.INDEX_FINGER_TIP

try:
    device = (
        AudioUtilities.GetDefaultOutputDevice()
        if hasattr(AudioUtilities, "GetDefaultOutputDevice")
        else AudioUtilities.GetSpeakers()
    )

    volume_control = device.EndpointVolume.QueryInterface(
        IAudioEndpointVolume
    )

    volume_min, volume_max = volume_control.GetVolumeRange()

except Exception as error:
    print(f"Audio control error: {error}")
    exit()


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam is not available.")
    exit()


WINDOW = "Gesture Based Control"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)


while True:

    success, frame = camera.read()

    if not success:
        break

    # Flip the frame for mirror-like display
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]

    # Process the frame using MediaPipe
    output = hand_detector.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )


    if output.multi_hand_landmarks and output.multi_handedness:

        for hand_index, hand_data in enumerate(
            output.multi_hand_landmarks
        ):

            hand_side = output.multi_handedness[
                hand_index
            ].classification[0].label

            # Draw hand landmarks
            drawing.draw_landmarks(
                frame,
                hand_data,
                HandModule.HAND_CONNECTIONS
            )

            landmarks = hand_data.landmark

            # Get thumb and index finger positions
            thumb_point = (
                int(landmarks[THUMB].x * frame_width),
                int(landmarks[THUMB].y * frame_height)
            )

            index_point = (
                int(landmarks[INDEX].x * frame_width),
                int(landmarks[INDEX].y * frame_height)
            )


            # Mark thumb and index finger
            cv2.circle(
                frame,
                thumb_point,
                9,
                (0, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                index_point,
                9,
                (0, 0, 255),
                cv2.FILLED
            )


            # Draw line between thumb and index finger
            cv2.line(
                frame,
                thumb_point,
                index_point,
                (255, 255, 0),
                3
            )


            # Calculate distance between thumb and index finger
            finger_distance = float(
                np.hypot(
                    index_point[0] - thumb_point[0],
                    index_point[1] - thumb_point[1]
                )
            )


            if hand_side == "Left":
                # Real RIGHT hand -> control volume

                current_volume = np.interp(
                    finger_distance,
                    [40, 280],
                    [volume_min, volume_max]
                )

                try:
                    volume_control.SetMasterVolumeLevel(
                        current_volume,
                        None
                    )

                except Exception as error:
                    print(f"Volume control error: {error}")


                volume_bar = int(
                    np.interp(
                        finger_distance,
                        [40, 280],
                        [390, 160]
                    )
                )

                volume_percent = int(
                    np.interp(
                        finger_distance,
                        [40, 280],
                        [0, 100]
                    )
                )


                # Draw volume bar
                cv2.rectangle(
                    frame,
                    (45, 160),
                    (80, 390),
                    (0, 255, 255),
                    2
                )

                cv2.rectangle(
                    frame,
                    (45, volume_bar),
                    (80, 390),
                    (0, 255, 255),
                    cv2.FILLED
                )


                cv2.putText(
                    frame,
                    f"{volume_percent}%",
                    (35, 440),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )


            elif hand_side == "Right":
                # Real LEFT hand -> control brightness

                brightness_level = int(
                    np.interp(
                        finger_distance,
                        [40, 280],
                        [0, 100]
                    )
                )


                try:
                    sbc.set_brightness(brightness_level)

                except Exception as error:
                    print(
                        f"Brightness control error: {error}"
                    )


                brightness_bar = int(
                    np.interp(
                        finger_distance,
                        [40, 280],
                        [390, 160]
                    )
                )


                bar_left = frame_width - 80
                bar_right = frame_width - 45


                # Draw brightness bar
                cv2.rectangle(
                    frame,
                    (bar_left, 160),
                    (bar_right, 390),
                    (0, 255, 0),
                    2
                )

                cv2.rectangle(
                    frame,
                    (bar_left, brightness_bar),
                    (bar_right, 390),
                    (0, 255, 0),
                    cv2.FILLED
                )


                cv2.putText(
                    frame,
                    f"{brightness_level}%",
                    (frame_width - 105, 440),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )


    # Display the final frame
    cv2.imshow(WINDOW, frame)


    # Exit using ESC or Q
    pressed = cv2.waitKey(1) & 0xFF

    if pressed in (27, ord("q")):
        break


    # Exit if the window is closed
    try:
        if cv2.getWindowProperty(
            WINDOW,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    except cv2.error:
        break


# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
