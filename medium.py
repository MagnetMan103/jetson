#!/usr/bin/env python3
"""
bottle_cv_opencv_dnn.py - Bottle Detection using OpenCV DNN
For Jetson Nano with IMX219 Camera
"""

import cv2
import numpy as np
import time
import urllib.request
import os


def create_gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0,
):
    """GStreamer pipeline for IMX219 camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def download_yolo_model():
    """Download YOLOv3-tiny model files if not present"""
    
    model_dir = "/root/yolo_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Model files
    weights_path = f"{model_dir}/yolov3.weights"
    config_path = f"{model_dir}/yolov3.cfg"
    names_path = f"{model_dir}/coco.names"
    
    # Download if not exists
    if not os.path.exists(weights_path):
        print("Downloading YOLOv3 weights...")
        urllib.request.urlretrieve(
            "https://pjreddie.com/media/files/yolov3.weights",
            weights_path
        )
    
    if not os.path.exists(config_path):
        print("Downloading YOLOv3-tiny config...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            config_path
        )
    
    if not os.path.exists(names_path):
        print("Downloading COCO class names...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            names_path
        )
    
    return weights_path, config_path, names_path


def detect_bottles_opencv_dnn():
    """Main detection function using OpenCV DNN"""
    
    print("Setting up YOLO model...")
    
    # Download model files
    weights_path, config_path, names_path = download_yolo_model()
    
    # Load COCO class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Bottle is class 39 in COCO
    bottle_class_id = 39
    
    print("Loading YOLO network...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Get output layer names - fixed for different OpenCV versions
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    
    # Handle different OpenCV versions
    if isinstance(unconnected, np.ndarray):
        if unconnected.ndim == 1:
            output_layers = [layer_names[i - 1] for i in unconnected]
        else:
            output_layers = [layer_names[i[0] - 1] for i in unconnected]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected]
    
    print("Model loaded successfully!")
    print("Opening IMX219 camera...")
    
    # Create GStreamer pipeline
    gst_pipeline = create_gstreamer_pipeline()
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Failed to open IMX219 camera")
        return
    
    print("Camera opened successfully!")
    
    CONFIDENCE_THRESHOLD = 0.5
    
    print("\n" + "=" * 50)
    print("Bottle Detection Started - OpenCV DNN")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%")
    print("Press ESC to exit")
    print("=" * 50 + "\n")
    print("CUDA working:", cv2.cuda.getCudaEnabledDeviceCount()) 
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.05)
                continue
            
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Run detection
            detections = net.forward(output_layers)
            
            # Process detections
            boxes = []
            confidences = []
            bottles_found = []
            
            for output in detections:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter for bottles only
                    if class_id == bottle_class_id and confidence > CONFIDENCE_THRESHOLD:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        bottles_found.append((center_x, center_y, w, h))
            
            # Apply non-maximum suppression
            bottles_detected = 0
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
                
                # Handle different return types
                if len(indices) > 0:
                    if isinstance(indices, tuple):
                        indices = indices[0]
                    if isinstance(indices, np.ndarray) and indices.ndim == 2:
                        indices = indices.flatten()
                    
                    bottles_detected = len(indices)
                    
                    if bottles_detected > 0:
                        print(f"\n[DETECTION] {bottles_detected} bottle(s) found:")
                        
                        for i, idx in enumerate(indices):
                            x, y, w, h = boxes[idx]
                            confidence = confidences[idx]
                            center_x, center_y, _, _ = bottles_found[idx]
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"Bottle {confidence*100:.1f}%"
                            cv2.putText(frame, label, (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Print to console
                            print(f"  Bottle {i+1}: Center=({center_x}, {center_y}), "
                                  f"Size={w}x{h}px, Confidence={confidence*100:.1f}%")
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Add info overlay
            cv2.putText(frame, f"Bottles: {bottles_detected}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press ESC to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('YOLO Bottle Detection - Jetson Nano', frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                print("\nESC pressed. Exiting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed. Goodbye!")


if __name__ == "__main__":
    detect_bottles_opencv_dnn()

