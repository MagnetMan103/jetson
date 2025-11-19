import cv2

# Define the GStreamer pipeline
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=21/1, format=NV12 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=960, height=616, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

# Open the camera using the pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

