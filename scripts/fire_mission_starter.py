#!/usr/bin/env python3
"""
Fire Mission Starter Script

This script runs on boot, monitors a GPIO switch, and starts the firefighting
mission when the switch is triggered. It also provides LED feedback.
The script launches navigation components but does not send navigation goals,
allowing the existing teensy_bridge fire detection system to handle goal publishing.
"""

import os
import time
import subprocess
import signal
import sys
import Jetson.GPIO as GPIO

# GPIO Configuration
SWITCH_PIN = 18  # GPIO pin for the switch input
LED_PIN = 12     # GPIO pin for the LED output

# ROS environment setup
ROS_SETUP_CMD = "source /opt/ros/humble/setup.bash && " \
                "source ~/robot_ws/install/setup.bash"

# Launch commands - sequential launch of minimal nav and nav2 bringup
MINIMAL_NAV_CMD = "ros2 launch ieee_robotics stable_minimal_nav.launch.py"
NAV2_BRINGUP_CMD = "ros2 launch nav2_bringup navigation_launch.py"

class FireMissionStarter:
    def __init__(self):
        self.setup_gpio()
        self.mission_running = False
        self.processes = []  # Store multiple processes
        print("Fire Mission Starter initialized")

    def setup_gpio(self):
        """Set up GPIO pins for switch input and LED output"""
        GPIO.setmode(GPIO.BCM)
        
        # Set up switch with pull-down resistor
        # When switch is off, pin reads LOW (0)
        # When switch is on, pin reads HIGH (1)
        GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Set up LED pin as output
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)  # Start with LED off
        
        print(f"GPIO initialized: Switch on pin {SWITCH_PIN}, LED on pin {LED_PIN}")

    def blink_led(self, times=3, interval=0.2):
        """Blink the LED to indicate status"""
        for _ in range(times):
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(interval)

    def solid_led(self, state=True):
        """Set LED to solid on or off"""
        GPIO.output(LED_PIN, GPIO.HIGH if state else GPIO.LOW)

    def run_shell_command(self, command, shell=True):
        """Run a shell command and return the process"""
        full_cmd = f'{ROS_SETUP_CMD} && {command}'
        print(f"Executing: {command}")
        return subprocess.Popen(full_cmd, shell=shell, executable='/bin/bash', 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               preexec_fn=os.setsid)

    def start_mission(self):
        """Start the fire finding mission"""
        if self.mission_running:
            print("Mission already running")
            return
        
        print("Starting fire mission...")
        self.mission_running = True
        
        # Indicate mission start with LED
        self.blink_led(times=5, interval=0.1)
        self.solid_led(True)  # Keep LED on while mission is running
        
        # 1. Launch minimal navigation first
        print("Starting minimal navigation...")
        minimal_nav_process = self.run_shell_command(MINIMAL_NAV_CMD)
        self.processes.append(minimal_nav_process)
        
        # Wait for minimal navigation to initialize
        print("Waiting for minimal navigation to initialize...")
        time.sleep(25)  # Give time for minimal nav to initialize
        
        # 2. Launch Nav2 bringup
        print("Starting Nav2 bringup...")
        nav2_process = self.run_shell_command(NAV2_BRINGUP_CMD)
        self.processes.append(nav2_process)
        
        # Important: We don't send a navigation goal here - the teensy_bridge fire detection
        # system will handle detection and goal publishing
        print("Navigation components started - teensy_bridge will handle fire detection and goals")

    def stop_mission(self):
        """Stop the running mission"""
        if not self.mission_running:
            return
            
        print("Stopping mission...")
        
        # Kill all processes in reverse order (last started, first killed)
        for process in reversed(self.processes):
            if process:
                try:
                    # Send SIGTERM to entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Error stopping process: {e}")
        
        # Clear the process list
        self.processes = []
        
        self.mission_running = False
        self.solid_led(False)  # Turn off LED
        print("Mission stopped")

    def monitor_switch(self):
        """Main loop to monitor the switch"""
        last_state = GPIO.input(SWITCH_PIN)
        print("Starting switch monitoring...")
        
        # Initial LED flash to show the script is running
        self.blink_led()

        try:
            while True:
                # Read current switch state
                current_state = GPIO.input(SWITCH_PIN)  # Here we monitor GPIO pin 18
                
                # Check if switch has been flipped from OFF to ON
                if current_state and not last_state:
                    print("Switch turned ON")
                    self.start_mission()
                
                # Check if switch has been flipped from ON to OFF
                elif not current_state and last_state:
                    print("Switch turned OFF")
                    self.stop_mission()
                
                # Update last state
                last_state = current_state
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up GPIO and processes before exiting"""
        self.stop_mission()
        GPIO.cleanup()
        print("Cleanup complete")


if __name__ == "__main__":
    # Allow time for system to boot fully
    time.sleep(30)
    
    print("Starting Fire Mission Starter")
    mission_starter = FireMissionStarter()
    mission_starter.monitor_switch()
