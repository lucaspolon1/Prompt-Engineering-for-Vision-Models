"""
Simple YOLO object detection on a single image.
Usage: python yolo_detect.py <image_path> [model_name]

Usage:
    python detect.py dogs.jpg
    python detect.py dogs.jpg yolo11m.pt
"""

import sys
import os
from ultralytics import YOLO
import cv2

def detect_objects(image_path, model_name="yolo26m.pt", output_dir="outputs"):
    """
    Run YOLO object detection on a single image.
    
    Args:
        image_path: Path to input image
        model_name: YOLO model to use
        output_dir: Where to save output
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    # Run detection
    print(f"Running detection on {image_path}...")
    results = model(image_path, verbose=True)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    
    # Plot results and save
    img_out = results[0].plot()
    cv2.imwrite(output_path, img_out)
    
    print(f"\n✓ Detection complete!")
    print(f"✓ Output saved to: {output_path}")
    print(f"\nDetected objects:")
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls]
            print(f"  - {name}: {conf:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolo_detect.py <image_path> [model_name]")
        print("Example: python yolo_detect.py dog.jpg yolo11m.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "yolo11m.pt"
    
    detect_objects(image_path, model_name)
