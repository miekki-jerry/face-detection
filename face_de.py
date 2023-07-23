from mtcnn import MTCNN
import cv2

# Initialize the MTCNN
detector = MTCNN()

# To capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    # Read the frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect faces
    try:
        result = detector.detect_faces(frame)
    except Exception as e:
        print("Failed to detect faces in the image. Error: ", str(e))
        continue

    # Draw bounding box and keypoints for each detected face
    for person in result:
        bounding_box = person['box']
        keypoints = person['keypoints']

        cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)

        cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)

    # display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
