#!/usr/bin/env python3
"""
bottle_chase_controller.py - Autonomous Bottle Chase for Hexapod Robot
Integrates bottle detection with hexapod navigation and orientation control
"""

import cv2
import numpy as np
import time
import urllib.request
import os
import signal
import sys
import serial
import threading

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\nShutdown signal received...")
    running = False


class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


class HexapodController:
    """Controls hexapod movement and orientation"""
    
    def __init__(self, port='/dev/ttyACM0', baud=9600):
        self.arduino = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        
        self.current_yaw = None
        self.running = True
        self.pause_reading = False
        self.read_lock = threading.Lock()
        
        # Start background thread to read IMU
        self.read_thread = threading.Thread(target=self._read_serial, daemon=True)
        self.read_thread.start()
        
        # Wait for first IMU reading
        print("Waiting for IMU data...")
        while self.current_yaw is None:
            time.sleep(0.1)
        print(f"IMU initialized. Current yaw: {self.current_yaw:.2f}")
    
    def _read_serial(self):
        """Background thread to continuously read from Arduino"""
        while self.running:
            # Pause reading when motor commands are being sent
            if self.pause_reading:
                time.sleep(0.05)
                continue
                
            if self.arduino.in_waiting > 0:
                try:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line.startswith("IMU:"):
                        data = line.split(":")[1]
                        yaw, pitch, roll = data.split(",")
                        with self.read_lock:
                            self.current_yaw = float(yaw)
                    elif line.startswith("MOTOR:"):
                        msg = line.split(":", 1)[1]
                        print(f"{Colors.BLUE}[MOTOR] {msg}{Colors.RESET}")
                    elif line.startswith("SYSTEM:"):
                        msg = line.split(":", 1)[1]
                        print(f"{Colors.YELLOW}[SYSTEM] {msg}{Colors.RESET}")
                except Exception as e:
                    # Silently ignore parsing errors
                    pass
            time.sleep(0.01)
    
    def get_current_yaw(self):
        """Get the current yaw angle"""
        with self.read_lock:
            return self.current_yaw
    
    def send_motor_command(self, direction, amplitude):
        """
        Send a motor command and wait for completion
        
        Args:
            direction: "LEFT", "RIGHT", "FORWARD", or "BACKWARD"
            amplitude: Motor amplitude (10-100)
        """
        # Pause background reading
        self.pause_reading = True
        time.sleep(0.1)  # Give background thread time to pause
        
        # Clear any buffered data
        self.arduino.reset_input_buffer()
        
        # Send command
        command = f"{direction}:{amplitude}\n"
        print(f"{Colors.BLUE}-> Sending: {command.strip()}{Colors.RESET}")
        self.arduino.write(command.encode())
        
        # Wait for motor action to complete (5 second timeout)
        # Read any responses during this time
        start_time = time.time()
        motor_complete = False
        
        while time.time() - start_time < 5.0:
            if self.arduino.in_waiting > 0:
                try:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line:
                        if line.startswith("IMU:"):
                            # Update yaw even during motor action
                            data = line.split(":")[1]
                            yaw, pitch, roll = data.split(",")
                            with self.read_lock:
                                self.current_yaw = float(yaw)
                        elif line.startswith("MOTOR:"):
                            msg = line.split(":", 1)[1]
                            print(f"{Colors.BLUE}[MOTOR] {msg}{Colors.RESET}")
                            if "complete" in msg.lower():
                                motor_complete = True
                                break
                        elif line.startswith("SYSTEM:"):
                            msg = line.split(":", 1)[1]
                            print(f"{Colors.YELLOW}[SYSTEM] {msg}{Colors.RESET}")
                except:
                    pass
            time.sleep(0.05)
        
        if not motor_complete:
            print(f"{Colors.YELLOW}Warning: Motor timeout - assuming complete{Colors.RESET}")
        
        # Additional settling time after motor action
        print("Waiting for robot to settle...")
        time.sleep(1.0)
        
        # Resume background reading
        self.pause_reading = False
        
        # Give time for fresh IMU reading
        time.sleep(0.3)
    
    def normalize_angle(self, angle):
        """Normalize angle to -180 to 180 range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_turn_error(self, current, target):
        """Calculate shortest path error between current and target angle"""
        error = target - current
        return self.normalize_angle(error)
    
    def turn_to_angle(self, target_angle, tolerance=5.0, max_iterations=15):
        """Turn the hexapod to a target angle using adaptive control"""
        target_angle = self.normalize_angle(target_angle)
        
        print(f"{Colors.CYAN}Turning to {target_angle:.1f} degrees{Colors.RESET}")
        
        start_yaw = self.get_current_yaw()
        amplitude = 80
        iteration = 0
        previous_error = None
        
        while iteration < max_iterations:
            current_yaw = self.get_current_yaw()
            error = self.calculate_turn_error(current_yaw, target_angle)
            
            # Check if we've reached the target
            if abs(error) <= tolerance:
                print(f"{Colors.GREEN}Turn complete! Final yaw: {current_yaw:.1f}{Colors.RESET}")
                return True
            
            # Check if we overshot
            if previous_error is not None:
                if (previous_error > 0 and error < 0) or (previous_error < 0 and error > 0):
                    amplitude = max(amplitude // 2, 10)
            
            # Determine turn direction
            direction = "LEFT" if error > 0 else "RIGHT"
            
            # Send motor command
            self.send_motor_command(direction, amplitude)
            
            previous_error = error
            iteration += 1
        
        # Max iterations reached
        current_yaw = self.get_current_yaw()
        print(f"{Colors.YELLOW}Turn timeout. Final yaw: {current_yaw:.1f}{Colors.RESET}")
        return False
    
    def move_forward(self, amplitude=60):
        """Move hexapod forward one step"""
        self.send_motor_command("FORWARD", amplitude)
    
    def close(self):
        """Close the serial connection"""
        self.running = False
        time.sleep(0.1)
        self.arduino.close()


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


def calculate_bottle_angle(bottle_x, frame_width, camera_fov=60):
    """
    Calculate the angle to turn to face the bottle
    
    Args:
        bottle_x: X position of bottle center in pixels
        frame_width: Width of the frame in pixels
        camera_fov: Camera field of view in degrees (default 60)
    
    Returns:
        Angle in degrees (negative = left, positive = right)
    """
    # Calculate offset from center as a fraction
    frame_center = frame_width / 2
    offset_fraction = (bottle_x - frame_center) / frame_center
    
    # Convert to angle based on FOV
    angle = offset_fraction * (camera_fov / 2)
    
    return angle


def estimate_distance(bottle_width, frame_width):
    """
    Estimate relative distance to bottle based on its width
    
    Returns:
        'close', 'medium', or 'far'
    """
    width_ratio = bottle_width / frame_width
    
    if width_ratio > 0.15:  # More than 15% of frame width - STOP
        return 'close'
    elif width_ratio > 0.08:  # 8-15% of frame width - approach slowly
        return 'medium'
    else:  # Less than 8% - approach normally
        return 'far'


def main():
    global running
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
    print(f"{Colors.GREEN}Hexapod Bottle Chase Controller{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
    
    # Initialize components
    try:
        print(f"{Colors.CYAN}Initializing hexapod controller...{Colors.RESET}")
        hexapod = HexapodController()
        
        print(f"\n{Colors.CYAN}Initializing bottle detector...{Colors.RESET}")
        bottle_detector = BottleDetector()
        
        # Open camera
        print("Opening IMX219 camera...")
        gst_pipeline = create_gstreamer_pipeline()
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"{Colors.RED}Failed to open camera{Colors.RESET}")
            return
        
        print(f"{Colors.GREEN}Camera opened successfully!{Colors.RESET}\n")
        
        # Camera parameters
        frame_width = 640
        frame_height = 480
        camera_fov = 60  # Degrees
        
        # Target distance threshold
        target_close_threshold = 0.15  # Bottle width should be 15% of frame (stop sooner)
        
        print(f"{Colors.CYAN}Starting bottle chase...{Colors.RESET}")
        print(f"{Colors.CYAN}Control loop: Capture frame -> Detect -> Act -> Repeat{Colors.RESET}")
        print(f"{Colors.CYAN}This ensures actions are always based on fresh footage{Colors.RESET}\n")
        
        frames_without_bottle = 0
        max_frames_without_bottle = 5  # Reduced since we're now synchronous
        
        while running:
            # STEP 1: Capture ONE fresh frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # STEP 2: Run detection on this frame
            bottles = bottle_detector.detect_bottles(frame)
            
            # STEP 3: Make decision based on detection
            if bottles:
                # Take the first (most confident) bottle
                center_x, center_y, width, height, confidence = bottles[0]
                
                # Calculate angle to bottle
                angle_to_bottle = calculate_bottle_angle(center_x, frame_width, camera_fov)
                
                # Estimate distance
                distance = estimate_distance(width, frame_width)
                
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp}] {Colors.GREEN}BOTTLE DETECTED{Colors.RESET}")
                print(f"  Position: ({center_x}, {center_y})")
                print(f"  Size: {width}x{height}px ({width/frame_width*100:.1f}% of frame)")
                print(f"  Angle: {angle_to_bottle:.1f} degrees")
                print(f"  Distance: {distance}")
                print(f"  Confidence: {confidence*100:.1f}%")
                
                # Reset lost bottle counter
                frames_without_bottle = 0
                
                # STEP 4: Execute action based on detection
                
                # Check if we're close enough - STOP
                if distance == 'close':
                    print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
                    print(f"{Colors.GREEN}TARGET REACHED! Bottle is close enough.{Colors.RESET}")
                    print(f"{Colors.GREEN}Bottle fills {width/frame_width*100:.1f}% of frame (threshold: {target_close_threshold*100:.0f}%){Colors.RESET}")
                    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
                    print(f"{Colors.YELLOW}Waiting 5 seconds before continuing search...{Colors.RESET}\n")
                    time.sleep(5)
                    continue
                
                # Check if bottle is drifting toward edge of frame
                # This catches drift even if angle calculation is slightly off
                x_position_ratio = center_x / frame_width
                is_drifting_left = x_position_ratio < 0.35  # Bottle in left 35% of frame
                is_drifting_right = x_position_ratio > 0.65  # Bottle in right 35% of frame
                
                # Turn to face the bottle if:
                # 1. Angle error is significant (>5 degrees), OR
                # 2. Bottle is drifting toward edges of frame
                needs_recentering = abs(angle_to_bottle) > 5 or is_drifting_left or is_drifting_right
                
                if needs_recentering:
                    if is_drifting_left or is_drifting_right:
                        edge = "LEFT" if is_drifting_left else "RIGHT"
                        print(f"\n{Colors.YELLOW}WARNING: Bottle drifting toward {edge} edge of frame!{Colors.RESET}")
                        print(f"  Bottle X position: {center_x}px ({x_position_ratio*100:.1f}% of frame)")
                    
                    print(f"{Colors.CYAN}Bottle is {angle_to_bottle:.1f} degrees off center{Colors.RESET}")
                    print(f"{Colors.CYAN}Recentering bottle in frame...{Colors.RESET}")
                    
                    # Calculate target angle relative to current orientation
                    current_yaw = hexapod.get_current_yaw()
                    target_yaw = current_yaw + angle_to_bottle
                    
                    # Turn to face bottle
                    hexapod.turn_to_angle(target_yaw, tolerance=5.0)
                    print(f"{Colors.GREEN}Recenter complete. Capturing fresh frame...{Colors.RESET}\n")
                    
                    # Continue to next iteration to get fresh frame after turn
                    continue
                
                else:
                    # Bottle is centered, move forward
                    print(f"\n{Colors.GREEN}Bottle well-centered!{Colors.RESET}")
                    print(f"  Angle: {angle_to_bottle:.1f} degrees (within 5 degree tolerance)")
                    print(f"  X Position: {center_x}px ({x_position_ratio*100:.1f}% from left)")
                    print(f"{Colors.GREEN}Moving forward...{Colors.RESET}")
                    
                    # Adjust speed based on distance
                    if distance == 'far':
                        amplitude = 60  # Normal speed for far targets
                        print(f"  Distance: FAR - Using amplitude {amplitude}")
                    else:  # medium
                        amplitude = 40  # Slower for medium targets  
                        print(f"  Distance: MEDIUM - Using amplitude {amplitude}")
                    
                    hexapod.move_forward(amplitude)
                    print(f"{Colors.GREEN}Move complete. Capturing fresh frame...{Colors.RESET}\n")
                    
                    # Continue to next iteration to get fresh frame after move
                    continue
            
            else:
                # No bottle detected
                frames_without_bottle += 1
                
                if frames_without_bottle == 1:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] No bottle detected. Searching...")
                
                if frames_without_bottle >= max_frames_without_bottle:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {Colors.YELLOW}No bottle found after {max_frames_without_bottle} frames.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Waiting for bottle...{Colors.RESET}\n")
                    frames_without_bottle = 0  # Reset counter
                
                # Short delay when no bottle to avoid spinning CPU
                time.sleep(0.2)
        
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        if 'cap' in locals():
            cap.release()
        if 'hexapod' in locals():
            hexapod.close()
        print(f"{Colors.GREEN}Shutdown complete.{Colors.RESET}")


if __name__ == "__main__":
    main()
