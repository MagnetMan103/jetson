from controller import *
from objectcv import *

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
