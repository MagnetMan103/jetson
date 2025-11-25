import numpy as np
import time
import urllib.request
import os 
import signal
import sys
import serial
import threading
from consts import Colors

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

