#!/bin/bash
# Setup script for the Fire Mission Starter

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Fire Mission Starter...${NC}"

# Get username
USERNAME=$(whoami)
if [ "$USERNAME" == "root" ]; then
    echo -e "${RED}Please run this script as a non-root user with sudo privileges${NC}"
    exit 1
fi

# Create scripts directory if it doesn't exist
SCRIPTS_DIR="$HOME/robot_ws/src/ieee_robotics/scripts"
mkdir -p "$SCRIPTS_DIR"

# Copy the Python script to the scripts directory
echo "Copying fire_mission_starter.py to $SCRIPTS_DIR"
cp fire_mission_starter.py "$SCRIPTS_DIR/"
chmod +x "$SCRIPTS_DIR/fire_mission_starter.py"

# Update username in the service file
echo "Updating username in service file"
sed -i "s/YOUR_USERNAME/$USERNAME/g" fire_mission.service

# Install required packages
echo "Installing required packages"
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install Jetson.GPIO

# Copy and enable the systemd service
echo "Setting up systemd service"
sudo cp fire_mission.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fire_mission.service

echo -e "${GREEN}Setup complete!${NC}"
echo "The mission starter will run on boot and monitor the switch on GPIO pin 18."
echo "To start the service manually: sudo systemctl start fire_mission"
echo "To check status: sudo systemctl status fire_mission"
echo -e "${RED}IMPORTANT: You must reboot for the changes to take effect.${NC}"
