
from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Video file name
video_path = "vedio.mp4"

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: '{video_path}' file not found!")
    exit()

# Open video
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error: Video could not be opened.")
    exit()

# Get video details for saving
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video file
out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video Finished!")
        break

    # Object Detection + Tracking
    results = model.track(frame, persist=True)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save processed frame
    out.write(annotated_frame)

    # Display output
    cv2.imshow("Object Detection & Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Output saved as output_video.mp4")