#pip install mediapipe
#pip install opencv-python
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(1)
# 0: webcam BN
# 1: webcam

#print(webcam)
image = webcam.read()
print(image)



# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Draw
mp_draw = mp.solutions.drawing_utils

while True:
        # Read a frame from the webcam
    success, image = webcam.read()

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    results_hand = hands.process(image_rgb)

# Check if any hands were detected
    if results_hand.multi_hand_landmarks:
        for landmark_hand in results_hand.multi_hand_landmarks:
            
        # Loop through all the landmarks and draw them on the image
            for landmark in landmark_hand.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                mp_draw.draw_landmarks(image, landmark_hand, mp_hands.HAND_CONNECTIONS)
                
    cv2.imshow("Image",image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
