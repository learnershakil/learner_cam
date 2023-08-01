import cv2
import dlib

def recognize_face():
    # Load the face detection model from dlib
    detector = dlib.get_frontal_face_detector()

    # Start capturing video from the default camera (usually 0) or any specified camera index
    camera = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera feed
        ret, frame = camera.read()

        if not ret:
            break

        # Convert the frame to grayscale (dlib requires grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame using dlib
        faces = detector(gray_frame)

        # Draw rectangles around the detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Detected Faces', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
