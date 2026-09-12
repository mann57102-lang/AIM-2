import cv2

# Load pre-trained Haar Cascade Classifier for face detection
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Initialize video capture using webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    success, image = camera.read()

    if not success:
        print("Error: Unable to capture frame")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    detected_faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(40, 40)
    )

    # Draw rectangles around detected faces
    for (left, top, width, height) in detected_faces:
        cv2.rectangle(
            image,
            (left, top),
            (left + width, top + height),
            (0, 255, 0),
            2
        )

    # Display the number of detected faces
    text_font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        image,
        f'Face Count: {len(detected_faces)}',
        (15, 35),
        text_font,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Display the frame with face detection and count
    cv2.imshow('Face Detection and Counting', image)

    # Exit the loop when the 'e' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the webcam and close the window
camera.release()
cv2.destroyAllWindows()
