import cv2
import numpy as np
import time
import threading
import urllib.request
import os


class BottleDetector:
    """Handles bottle detection using YOLO"""
    
    def __init__(self):
        print("Setting up bottle detector...")
        
        # Download and load YOLO model
        weights_path, config_path, names_path = self.download_yolo_model()
        
        # Load COCO class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Bottle is class 39 in COCO
        self.bottle_class_id = 39
        
        # Load YOLO network
        print("Loading YOLO network...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        
        if isinstance(unconnected, np.ndarray):
            if unconnected.ndim == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        
        print("Bottle detector ready!")
        
        self.confidence_threshold = 0.5
    
    def download_yolo_model(self):
        """Download YOLOv3-tiny model files if not present"""
        model_dir = "/root/yolo_models"
        os.makedirs(model_dir, exist_ok=True)
        
        weights_path = f"{model_dir}/yolov3-tiny.weights"
        config_path = f"{model_dir}/yolov3-tiny.cfg"
        names_path = f"{model_dir}/coco.names"
        
        if not os.path.exists(weights_path):
            print("Downloading YOLOv3-tiny weights...")
            urllib.request.urlretrieve(
                "https://pjreddie.com/media/files/yolov3-tiny.weights",
                weights_path
            )
        
        if not os.path.exists(config_path):
            print("Downloading YOLOv3-tiny config...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                config_path
            )
        
        if not os.path.exists(names_path):
            print("Downloading COCO class names...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                names_path
            )
        
        return weights_path, config_path, names_path
    
    def detect_bottles(self, frame):
        """
        Detect bottles in a frame
        Returns: List of detected bottles with (center_x, center_y, width, height, confidence)
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run detection
        detections = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        bottles = []
        
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for bottles only
                if class_id == self.bottle_class_id and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    bottles.append((center_x, center_y, w, h, confidence))
        
        # Apply non-maximum suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                if isinstance(indices, np.ndarray) and indices.ndim == 2:
                    indices = indices.flatten()
                
                # Return filtered bottles
                return [bottles[i] for i in indices]
        
        return []

