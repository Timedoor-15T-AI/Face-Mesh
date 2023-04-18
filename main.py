import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) # 0 for webcam, 1 for external camera
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
            mpDraw.draw_landmarks( # Draw the landmarks
                image = frame,
                landmark_list = face_landmarks,
                connections = mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mpDrawingStyle.get_default_face_mesh_tesselation_style()
            )

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()