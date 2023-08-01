import cv2

def recognize_face(cascade_path):
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Start capturing video from the default camera (usually 0) or any specified camera index
    camera = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera feed
        ret, frame = camera.read()

        if not ret:
            break

        # Convert the frame to grayscale (required for the Haar Cascade Classifier)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
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
    cascade_path = 'path_to_haar_cascade.xml'  # Replace with the path to the Haar Cascade XML file
    recognize_face(cascade_path)
