# faceDetection
#Python program that detects a face and check if that face is avaible in the database if not available ask the user to enter the details and the details will be #added in the sql database
import cv2
import face_recognition
import mysql.connector
import numpy as np

# Establish a connection to the MySQL database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="face_recognition"
)

# Function to encode a face image
def encode_face(image):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use face_recognition to locate faces in the image
    boxes = face_recognition.face_locations(rgb, model='hog')
    # Encode the faces and store the encodings in a list
    encodings = face_recognition.face_encodings(rgb, boxes)
    # Return the first encoding if a face was found, else return None
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None

# Function to compare a face encoding with a list of known encodings
def compare_faces(encoding, encodings):
    # Compare the input encoding with a list of known encodings to see if there is a match
    matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.6)
    # Return the index of the first match, or None if no match was found
    match_index = np.argmax(matches) if any(matches) else None
    return match_index

# Function to recognize faces in real-time using a webcam
def recognize_face():
    # Create a cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Open the video capture device (webcam)
    video_capture = cv2.VideoCapture(0)
    # Lists to store known encodings, names, and ages
    known_encodings = []
    known_names = []
    known_ages = []
    # Create a cursor to execute SQL queries
    cursor = mydb.cursor()
    # Retrieve data from the "users" table in the database
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    # Iterate over the retrieved rows
    for row in results:
        # Extract the face encoding, name, and age from each row
        face_encoding = np.fromstring(row[3], dtype=float, sep=',')
        name = row[1]
        age = row[2]
        # Append the encoding, name, and age to the respective lists
        known_encodings.append(face_encoding)
        known_names.append(name)
        known_ages.append(age)
    # Start the main loop for real-time face recognition
    while True:
        # Read a frame from the video capture device
        _, frame = video_capture.read()
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around each face in the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 100, 200), 5)
            # Extract the region of interest (face) from the grayscale frame
            roi_gray = gray[y:y+h, x:x+w]
            # Extract the region of interest (face) from the color frame
            roi_color = frame[y:y+h, x:x+w]
            # Encode the face
            encoding = encode_face(roi_color)
            # Check if encoding is not None (a face was found)
            if encoding is not None:
                # Compare the encoding with the known encodings
                match_index = compare_faces(encoding, known_encodings)
                # Check if a match was found
                if match_index is not None:
                    # Retrieve the name and age of the matched face
                    name = known_names[match_index]
                    age = known_ages[match_index]
                    # Display the name and age on the frame
                    cv2.putText(frame, f"Name: {name}, Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Face not recognized, prompt user to add it to the database
                    name = input("Enter your name: ")
                    age = input("Enter your age: ")
                    # Convert the face encoding to a string for storage in the database
                    encoding_str = ','.join(map(str, encoding))
                    # Insert the new user into the "users" table in the database
                    sql = "INSERT INTO users (name, age, face_encoding) VALUES (%s, %s, %s)"
                    val = (name, age, encoding_str)
                    cursor.execute(sql, val)
                    mydb.commit()
                    print("Face added to database")
                    # Append the new encoding, name, and age to the respective lists
                    known_encodings.append(encoding)
                    known_names.append(name)
                known_ages.append(age)
                # Display the name and age on the frame
                cv2.putText(frame, f"Name: {name}, Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Display the frame
        cv2.imshow('Video', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture device and close the windows
    video_capture.release()
    cv2.destroyAllWindows()

# Call the recognize_face function to start face recognition
recognize_face()
