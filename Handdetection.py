import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    ret, image = cap.read()
    
    if not ret:
        print("Failed to capture frame from the camera")
        break

    # Ensure the image has 3 channels (BGR)
    if image.shape[2] != 3:
        continue  # Skip this frame if the image doesn't have 3 channels

    # Flip the image
    image = cv2.flip(image, 1)
    
    # Storing the results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            )
    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
