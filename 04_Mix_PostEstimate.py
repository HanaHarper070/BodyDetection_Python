#04_Mix_PostEstimate.py
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(1)
# 0: webcam BN
# 1: webcam

#print(webcam)
image = webcam.read()
#print(image)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe Face model
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)

# Initialize MediaPipe Pose Estimate model
mp_Pose = mp.solutions.pose
pose = mp_Pose.Pose()

# Draw
mp_draw = mp.solutions.drawing_utils

while True:
# Read a frame from the webcam
    success, image = webcam.read()

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    results_hand = hands.process(image_rgb)
    
    results_faceMesh = faceMesh.process(image_rgb)
    
    results_pose = pose.process(image_rgb)

# Check if any hands were detected
    if results_hand.multi_hand_landmarks:
        for landmark_hand in results_hand.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, landmark_hand, mp_hands.HAND_CONNECTIONS)
          

# Check if any faces were detected
    if results_faceMesh.multi_face_landmarks:
        for landmark_face in results_faceMesh.multi_face_landmarks:
            mp_draw.draw_landmarks(image, landmark_face, mpFaceMesh.FACEMESH_TESSELATION)

# Check if any hands were detected
    if results_pose.pose_landmarks:
        #for landmark_pose in results_pose.pose_landmarks:
        mp_draw.draw_landmarks(image, results_pose.pose_landmarks, mp_Pose.POSE_CONNECTIONS)


    cv2.imshow("Image",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break