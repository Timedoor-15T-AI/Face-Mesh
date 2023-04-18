import cv2
import mediapipe as mp
from math import hypot

cap = cv2.VideoCapture(0) # 0 for webcam, 1 for external camera
# load pig nose image
nose_image = cv2.imread("pig_nose.png")
cap.set(3, 640) # Width
cap.set(4, 480) # Height

# Add instance for mediapipe
mpDraw = mp.solutions.drawing_utils # For drawing the landmarks
mpDrawingStyle = mp.solutions.drawing_styles # For drawing the landmarks
# Add instance for face mesh
mpFaceMesh = mp.solutions.face_mesh # For face mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 4) # For face mesh

while True:
    ret, frame = cap.read()

    # Add code for face mesh
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
    results = faceMesh.process(rgb) # Process the image 

    if results.multi_face_landmarks: # If there is a face
        for face_landmarks in results.multi_face_landmarks: # For each face
            
            leftnoseX = 0
            leftnoseY = 0
            rightnoseX = 0
            rightnoseY = 0
            centernoseX = 0
            centernoseY = 0

            for lm_id, lm in enumerate(face_landmarks.landmark): # For each landmark
                h, w, c = frame.shape # Get the height, width and channels
                x, y = int(lm.x * w), int(lm.y * h) # Get the x and y coordinates

                if lm_id == 49: # Left nose
                    leftnoseX, leftnoseY = x, y

                if lm_id == 279: # Right nose
                    rightnoseX, rightnoseY = x, y

                if lm_id == 5: # Center nose
                    centernoseX, centernoseY = x, y

            # Add nose image
            nose_width = int(hypot(rightnoseX - leftnoseX, rightnoseY - leftnoseY) * 1.2)
            nose_height = int(nose_width * 0.77)
            # New nose position and size
            if (nose_width and nose_height) != 0:
                pig_nose = cv2.resize(nose_image, (nose_width, nose_height))

            # Find coordinates for nose
            top_left = (int(centernoseX - nose_width / 2), int(centernoseY - nose_height / 2))
            bottom_right = (int(centernoseX + nose_width / 2), int(centernoseY + nose_height / 2))
            nose_area = frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] # Get the nose area

            # Add the nose image to the nose area
            pig_nose_gray = cv2.cvtColor(pig_nose, cv2.COLOR_BGR2GRAY) # Convert to gray
            _, nose_mask = cv2.threshold(pig_nose_gray, 25, 255, cv2.THRESH_BINARY_INV) # Get the mask
            no_nose = cv2.bitwise_and(nose_area, nose_area, mask = nose_mask) # Remove the nose from the nose area
            final_nose = cv2.add(no_nose, pig_nose) # Add the nose to the nose area
            frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose # Add the nose area to the frame

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()