import cv2


# Open three webcam feeds
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(4)

for cap in [cap1, cap2, cap3]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read frames from each camera
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if not (ret1 and ret2 and ret3):
        print("Failed to grab one or more frames")
        break

    # Display the frames in a single window
    combined_frame = cv2.hconcat([frame1, frame2, frame3])
    cv2.imshow('Webcam Feeds', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close the window
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()
