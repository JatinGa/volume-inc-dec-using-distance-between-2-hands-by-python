import cv2
import mediapipe as mp
import platform
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Function to set the system volume for Windows
def set_system_volume_windows(volume):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
    volume_interface.SetMasterVolumeLevelScalar(volume / 100.0, None)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Extract the landmarks for both hands
        if len(results.multi_hand_landmarks) >= 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            # Get coordinates for the tip of the index fingers (landmark 8)
            x1, y1 = hand1.landmark[8].x, hand1.landmark[8].y
            x2, y2 = hand2.landmark[8].x, hand2.landmark[8].y

            # Calculate the distance between the index fingers
            distance = calculate_distance((x1, y1), (x2, y2))
            print(f"Distance: {distance}")

            # Convert distance to a volume level
            max_distance = 0.4  # Adjust this value based on your setup
            max_volume = 100
            volume_level = max_volume * (1 - min(distance / max_distance, 1))
            volume_level = round(volume_level)
            print(f"Calculated Volume Level: {volume_level}")

            # Set system volume for Windows
            set_system_volume_windows(volume_level)

            # Draw the hand annotations on the image
            mp_drawing.draw_landmarks(image, hand1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, hand2, mp_hands.HAND_CONNECTIONS)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
