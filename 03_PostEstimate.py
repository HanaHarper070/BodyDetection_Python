#03_PostEstimate.py
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(1)
# 0: webcam BN
# 1: webcam

#print(webcam)
image = webcam.read()
#print(image)

# Initialize MediaPipe Pose Estimate model
mp_Pose = mp.solutions.pose
pose = mp_Pose.Pose()

# Draw
mp_draw = mp.solutions.drawing_utils

while True:
        # Read a frame from the webcam
    success, image = webcam.read()

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    results_pose = pose.process(image_rgb)

# Check if any hands were detected
    if results_pose.pose_landmarks:
        #for landmark_pose in results_pose.pose_landmarks:
        mp_draw.draw_landmarks(image, results_pose.pose_landmarks, mp_Pose.POSE_CONNECTIONS)


    cv2.imshow("Image",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break