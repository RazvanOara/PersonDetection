import torch
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', device=device)  # Load YOLOv5 large model for better accuracy
model.eval()  # Set the model to evaluation mode

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

def load_image(image_path_or_url):
    try:
        img = Image.open(image_path_or_url).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def detect_persons(img):
    """Perform detection and return results."""
    try:
        results = model(img)  # Perform detection
        detections = results.pandas().xyxy[0]  # Results in a DataFrame
        print("Detections:", detections)  # Print detections for debugging
        return detections
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def draw_pose(img, detections):
    """Draw skeleton on detected persons."""
    img_height, img_width, _ = img.shape

    for _, row in detections.iterrows():
        if row['class'] == 0:  # Class ID for 'person' in COCO dataset
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            roi = img[y1:y2, x1:x2]

            # Convert ROI to RGB and process with MediaPipe Pose
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(roi_rgb)

            if results.pose_landmarks:
                # Draw landmarks and connections
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_landmark = results.pose_landmarks.landmark[start_idx]
                    end_landmark = results.pose_landmarks.landmark[end_idx]

                    # Convert normalized coordinates to ROI coordinates
                    start_x = int(start_landmark.x * (x2 - x1) + x1)
                    start_y = int(start_landmark.y * (y2 - y1) + y1)
                    end_x = int(end_landmark.x * (x2 - x1) + x1)
                    end_y = int(end_landmark.y * (y2 - y1) + y1)

                    # Draw line between the landmarks
                    cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    return img

def visualize_results(image_path_or_url, detections, resize_factor=0.5):
    """Visualize results on image with optional resizing."""
    try:
        print("Visualizing results...")
        img = cv2.imread(image_path_or_url)
        img = draw_pose(img, detections)

        # Optional resizing before display
        height, width = img.shape[:2]
        new_height = int(height * resize_factor)
        new_width = int(width * resize_factor)
        img = cv2.resize(img, (new_width, new_height))

        # Display image with OpenCV
        cv2.imshow("Detection Results (Resized)", img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Destroy the window
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    # Define the image path (local file path)
    image_path_or_url = r'D:\Facultate\Practica\PersonDetection\local_image3.jpg'  # Replace with your local image path

    # Load and process image
    img = load_image(image_path_or_url)
    if img is not None:
        # Detect persons
        detections = detect_persons(img)
        if detections is not None:
            # Visualize results
            visualize_results(image_path_or_url, detections)
