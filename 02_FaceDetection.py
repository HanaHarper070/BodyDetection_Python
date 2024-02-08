#02_FaceDetection.py
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(1)
# 0: webcam BN
# 1: webcam

#print(webcam)
image = webcam.read()
#print(image)

# Initialize MediaPipe Face model
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)

# Draw
mp_draw = mp.solutions.drawing_utils

while True:
        # Read a frame from the webcam
    success, image = webcam.read()

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    results_faceMesh = faceMesh.process(image_rgb)
    # print(results_faceMesh.multi_face_landmarks)


# Check if any hands were detected
    if results_faceMesh.multi_face_landmarks:
        for landmark_face in results_faceMesh.multi_face_landmarks:
            mp_draw.draw_landmarks(image, landmark_face, mpFaceMesh.FACEMESH_TESSELATION)


    cv2.imshow("Image",image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
