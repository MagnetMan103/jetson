from controller import *
from objectcv import *
from consts import Colors
from camera import FrameBuffer

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
    frame_center = frame_width / 2
    offset_fraction = (bottle_x - frame_center) / frame_center
    angle = offset_fraction * (camera_fov / 2)
    return angle


def estimate_distance(bottle_width, frame_width):
    """
    Estimate relative distance to bottle based on its width

    Returns:
        'close', 'medium', or 'far'
    """
    width_ratio = bottle_width / frame_width

    if width_ratio > 0.15:
        return 'close'
    elif width_ratio > 0.08:
        return 'medium'
    else:
        return 'far'


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)

    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
    print(f"{Colors.GREEN}Hexapod Bottle Chase Controller{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")

    fb = None
    try:
        print(f"{Colors.CYAN}Initializing hexapod controller...{Colors.RESET}")
        hexapod = HexapodController()

        print(f"\n{Colors.CYAN}Initializing bottle detector...{Colors.RESET}")
        bottle_detector = BottleDetector()

        print("Opening IMX219 camera...")
        fb = FrameBuffer(
            use_gstreamer=True,
            sensor_id=0,
            capture_width=1640,
            capture_height=1232,
            display_width=640,
            display_height=480,
            framerate=21,
            flip_method=0,
            buffer_size=1
        )

        print(f"{Colors.GREEN}Camera opened successfully!{Colors.RESET}\n")

        frame_width = 640
        frame_height = 480
        camera_fov = 60
        target_close_threshold = 0.15

        print(f"{Colors.CYAN}Starting bottle chase...{Colors.RESET}")
        print(f"{Colors.CYAN}Control loop: Capture frame -> Detect -> Act -> Repeat{Colors.RESET}")
        print(f"{Colors.CYAN}This ensures actions are always based on fresh footage{Colors.RESET}\n")

        frames_without_bottle = 0
        max_frames_without_bottle = 5

        while running:
            frame = fb.get_latest_frame(copy=True)

            if frame is None:
                time.sleep(0.1)
                continue

            bottles = bottle_detector.detect_bottles(frame)

            if bottles:
                center_x, center_y, width, height, confidence = bottles[0]
                angle_to_bottle = calculate_bottle_angle(center_x, frame_width, camera_fov)
                distance = estimate_distance(width, frame_width)

                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp}] {Colors.GREEN}BOTTLE DETECTED{Colors.RESET}")
                print(f"  Position: ({center_x}, {center_y})")
                print(f"  Size: {width}x{height}px ({width/frame_width*100:.1f}% of frame)")
                print(f"  Angle: {angle_to_bottle:.1f} degrees")
                print(f"  Distance: {distance}")
                print(f"  Confidence: {confidence*100:.1f}%")

                frames_without_bottle = 0

                if distance == 'close':
                    print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
                    print(f"{Colors.GREEN}TARGET REACHED! Bottle is close enough.{Colors.RESET}")
                    print(f"{Colors.GREEN}Bottle fills {width/frame_width*100:.1f}% of frame (threshold: {target_close_threshold*100:.0f}%){Colors.RESET}")
                    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
                    print(f"{Colors.YELLOW}Waiting 5 seconds before continuing search...{Colors.RESET}\n")
                    time.sleep(5)
                    continue

                x_position_ratio = center_x / frame_width
                is_drifting_left = x_position_ratio < 0.35
                is_drifting_right = x_position_ratio > 0.65
                needs_recentering = abs(angle_to_bottle) > 5 or is_drifting_left or is_drifting_right

                if needs_recentering:
                    if is_drifting_left or is_drifting_right:
                        edge = "LEFT" if is_drifting_left else "RIGHT"
                        print(f"\n{Colors.YELLOW}WARNING: Bottle drifting toward {edge} edge of frame!{Colors.RESET}")
                        print(f"  Bottle X position: {center_x}px ({x_position_ratio*100:.1f}% of frame)")

                    print(f"{Colors.CYAN}Bottle is {angle_to_bottle:.1f} degrees off center{Colors.RESET}")
                    print(f"{Colors.CYAN}Recentering bottle in frame...{Colors.RESET}")

                    current_yaw = hexapod.get_current_yaw()
                    target_yaw = current_yaw + angle_to_bottle

                    hexapod.turn_to_angle(target_yaw, tolerance=5.0)
                    print(f"{Colors.GREEN}Recenter complete. Capturing fresh frame...{Colors.RESET}\n")
                    continue

                else:
                    print(f"\n{Colors.GREEN}Bottle well-centered!{Colors.RESET}")
                    print(f"  Angle: {angle_to_bottle:.1f} degrees (within 5 degree tolerance)")
                    print(f"  X Position: {center_x}px ({x_position_ratio*100:.1f}% from left)")
                    print(f"{Colors.GREEN}Moving forward...{Colors.RESET}")

                    if distance == 'far':
                        amplitude = 60
                        print(f"  Distance: FAR - Using amplitude {amplitude}")
                    else:
                        amplitude = 40
                        print(f"  Distance: MEDIUM - Using amplitude {amplitude}")

                    hexapod.move_forward(amplitude)
                    print(f"{Colors.GREEN}Move complete. Capturing fresh frame...{Colors.RESET}\n")
                    continue

            else:
                frames_without_bottle += 1

                if frames_without_bottle == 1:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] No bottle detected. Searching...")

                if frames_without_bottle >= max_frames_without_bottle:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {Colors.YELLOW}No bottle found after {max_frames_without_bottle} frames.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Waiting for bottle...{Colors.RESET}\n")
                    frames_without_bottle = 0

                time.sleep(0.2)

    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        if fb is not None:
            fb.stop()
        if 'hexapod' in locals():
            hexapod.close()
        print(f"{Colors.GREEN}Shutdown complete.{Colors.RESET}")


if __name__ == "__main__":
    main()
