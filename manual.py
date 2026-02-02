#!/usr/bin/env python3
"""
Hexapod Manual Control Script
Simple keyboard control for hexapod robot

Commands:
  1 [amplitude] - Move FORWARD
  2 [amplitude] - Move BACKWARD  
  3 [amplitude] - Turn LEFT
  4 [amplitude] - Turn RIGHT
  q - Quit

Default amplitude: 50
Example: "1 70" moves forward with amplitude 70
"""

import serial
import time
import sys


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


class HexapodManualController:
    """Simple manual controller for hexapod robot"""
    
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        """Initialize serial connection to Arduino"""
        print(f"{Colors.CYAN}Connecting to hexapod on {port}...{Colors.RESET}")
        
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"{Colors.GREEN}Connected successfully!{Colors.RESET}\n")
        except serial.SerialException as e:
            print(f"{Colors.RED}Failed to connect: {e}{Colors.RESET}")
            sys.exit(1)
    
    def send_command(self, direction, amplitude):
        """
        Send a motor command to the hexapod
        
        Args:
            direction: "FORWARD", "BACKWARD", "LEFT", or "RIGHT"
            amplitude: Motor amplitude (10-100)
        """
        # Clamp amplitude to valid range
        amplitude = max(10, min(100, amplitude))
        
        # Clear any buffered data
        self.arduino.reset_input_buffer()
        
        # Send command
        command = f"{direction}:{amplitude}\n"
        print(f"{Colors.BLUE}Sending: {direction} with amplitude {amplitude}{Colors.RESET}")
        self.arduino.write(command.encode())
        
        # Wait for motor action to complete (5 second timeout)
        start_time = time.time()
        motor_complete = False
        
        while time.time() - start_time < 5.0:
            if self.arduino.in_waiting > 0:
                try:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line:
                        if line.startswith("MOTOR:"):
                            msg = line.split(":", 1)[1]
                            print(f"{Colors.BLUE}[MOTOR] {msg}{Colors.RESET}")
                            if "complete" in msg.lower():
                                motor_complete = True
                                break
                        elif line.startswith("SYSTEM:"):
                            msg = line.split(":", 1)[1]
                            print(f"{Colors.YELLOW}[SYSTEM] {msg}{Colors.RESET}")
                except Exception as e:
                    pass
            time.sleep(0.05)
        
        if not motor_complete:
            print(f"{Colors.YELLOW}Warning: Motor timeout - assuming complete{Colors.RESET}")
        
        print(f"{Colors.GREEN}Movement complete!{Colors.RESET}\n")
    
    def close(self):
        """Close the serial connection"""
        self.arduino.close()
        print(f"{Colors.GREEN}Connection closed.{Colors.RESET}")


def print_instructions():
    """Print usage instructions"""
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}Hexapod Manual Control{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}\n")
    print(f"{Colors.CYAN}Commands:{Colors.RESET}")
    print(f"  {Colors.YELLOW}1 [amplitude]{Colors.RESET} - Move FORWARD")
    print(f"  {Colors.YELLOW}2 [amplitude]{Colors.RESET} - Move BACKWARD")
    print(f"  {Colors.YELLOW}3 [amplitude]{Colors.RESET} - Turn LEFT")
    print(f"  {Colors.YELLOW}4 [amplitude]{Colors.RESET} - Turn RIGHT")
    print(f"  {Colors.YELLOW}q{Colors.RESET} - Quit")
    print(f"\n{Colors.CYAN}Default amplitude: 50{Colors.RESET}")
    print(f"{Colors.CYAN}Amplitude range: 10-100{Colors.RESET}")
    print(f"\n{Colors.CYAN}Examples:{Colors.RESET}")
    print(f"  1     - Move forward with amplitude 50")
    print(f"  1 70  - Move forward with amplitude 70")
    print(f"  3 30  - Turn left with amplitude 30")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}\n")


def parse_command(user_input):
    """
    Parse user input into direction and amplitude
    
    Returns:
        (direction, amplitude) tuple or (None, None) if invalid
    """
    parts = user_input.strip().split()
    
    if not parts:
        return None, None
    
    # Command mapping
    command_map = {
        '1': 'FORWARD',
        '2': 'BACKWARD',
        '3': 'LEFT',
        '4': 'RIGHT'
    }
    
    command = parts[0].lower()
    
    # Check for quit command
    if command == 'q':
        return 'QUIT', None
    
    # Check for valid direction command
    if command not in command_map:
        print(f"{Colors.RED}Invalid command: {command}{Colors.RESET}")
        print(f"{Colors.YELLOW}Use 1/2/3/4 for direction or 'q' to quit{Colors.RESET}\n")
        return None, None
    
    direction = command_map[command]
    
    # Parse amplitude if provided
    default_amplitude = 50
    amplitude = default_amplitude
    
    if len(parts) > 1:
        try:
            amplitude = int(parts[1])
            if amplitude < 10 or amplitude > 100:
                print(f"{Colors.YELLOW}Warning: Amplitude {amplitude} out of range (10-100), clamping...{Colors.RESET}")
                amplitude = max(10, min(100, amplitude))
        except ValueError:
            print(f"{Colors.YELLOW}Invalid amplitude: {parts[1]}, using default {default_amplitude}{Colors.RESET}")
            amplitude = default_amplitude
    
    return direction, amplitude


def main():
    """Main control loop"""
    print_instructions()
    
    # Initialize hexapod controller
    controller = HexapodManualController()
    
    try:
        print(f"{Colors.CYAN}Ready for commands! (Type 'q' to quit){Colors.RESET}\n")
        
        while True:
            # Get user input
            try:
                user_input = input(f"{Colors.GREEN}Command > {Colors.RESET}")
            except EOFError:
                break
            
            # Parse command
            direction, amplitude = parse_command(user_input)
            
            # Handle quit command
            if direction == 'QUIT':
                print(f"\n{Colors.CYAN}Exiting...{Colors.RESET}")
                break
            
            # Send valid commands
            if direction is not None and amplitude is not None:
                controller.send_command(direction, amplitude)
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Interrupted by user{Colors.RESET}")
    
    finally:
        # Cleanup
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        controller.close()
        print(f"{Colors.GREEN}Goodbye!{Colors.RESET}")


if __name__ == "__main__":
    main()
