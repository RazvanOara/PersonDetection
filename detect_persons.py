import torch
import cv2
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # Load YOLOv5 small model and move to device
model.eval()  # Set the model to evaluation mode

def load_image(image_path_or_url):
        img = Image.open(image_path_or_url).convert('RGB')
        return img

def detect_persons(img):
    """Perform detection and return results."""
    results = model(img)  # Perform detection
    detections = results.pandas().xyxy[0]  # Results in a DataFrame
    print("Detections:", detections)  # Print detections for debugging
    return detections

def visualize_results(image_path_or_url, detections, resize_factor=0.5):
    """Visualize results on image with optional resizing."""
    print("Visualizing results...")
    # Load image from local path
    img = cv2.imread(image_path_or_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    # Filter detections for persons
    for _, row in detections.iterrows():
        if row['class'] == 0:  # Class ID for 'person' in COCO dataset
            x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(img, f'Person: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Optional resizing before display
    height, width = img.shape[:2]
    new_height = int(height * resize_factor)
    new_width = int(width * resize_factor)
    img = cv2.resize(img, (new_width, new_height))

    # Display image with OpenCV
    cv2.imshow("Detection Results (Resized)", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Destroy the window

if __name__ == "__main__":
    # Define the image path (local file path)
    image_path_or_url = r'D:\Facultate\Practica\PersonDetection\local_image3.jpg'  # Replace with your local image path

    # Load and process image
    img = load_image(image_path_or_url)

    # Detect persons
    detections = detect_persons(img)

    # Visualize results
    visualize_results(image_path_or_url, detections)
