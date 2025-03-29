#!/bin/bash
# GPU Optimization Script for Jetson Xavier NX
# Save this file to ~/robot_ws/src/ieee_robotics/scripts/jetson_gpu_setup.sh

echo "Optimizing Jetson Xavier NX for GPU acceleration in robotics tasks..."

# 1. Set maximum performance mode with GPU priority
echo "Setting Jetson to MAXN performance mode with GPU priority..."
sudo nvpmodel -m 0  # MAXN mode with maximum performance
sudo jetson_clocks   # Set CPU, GPU, and EMC clocks to maximum

# 2. Configure CUDA environment variables
echo "Setting up CUDA environment..."
cat >> ~/.bashrc << 'EOF'

# CUDA and GPU Environment
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDNN_PATH=/usr/lib/aarch64-linux-gnu
export OPENBLAS_CORETYPE=ARMV8

# Set OpenCV to use CUDA
export OPENCV_DNN_CUDA=ON

# Set thread priority for ROS 2 real-time operations
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=$HOME/ros2_cyclonedds_config/cyclonedds.xml
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_COLORIZED_OUTPUT=1
EOF

source ~/.bashrc

# 3. Make sure necessary packages are installed
echo "Checking for required packages..."
sudo apt update
packages=(
    "libopencv-dev"
    "ros-humble-cv-bridge"
    "python3-opencv"
)

for pkg in "${packages[@]}"; do
    if ! dpkg -l | grep -q $pkg; then
        echo "Installing $pkg..."
        sudo apt install -y $pkg
    else
        echo "$pkg is already installed."
    fi
done

# 4. Check if we have GPU-enabled OpenCV
echo "Checking OpenCV CUDA support..."
python3 -c "import cv2; print('OpenCV CUDA Support:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# 5. Create CycloneDDS config optimized for Jetson
echo "Creating optimized CycloneDDS config..."
mkdir -p ~/ros2_cyclonedds_config
cat > ~/ros2_cyclonedds_config/cyclonedds.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain>
        <General>
            <NetworkInterfaceAddress>auto</NetworkInterfaceAddress>
            <AllowMulticast>true</AllowMulticast>
            <MaxMessageSize>65500B</MaxMessageSize>
        </General>
        <Internal>
            <Watermarks>
                <WhcHigh>8MB</WhcHigh>
            </Watermarks>
        </Internal>
        <Tracing>
            <Verbosity>warning</Verbosity>
            <OutputFile>/tmp/cyclonedds.log</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>
EOF

echo "Configuring system for optimal performance..."
# Adjust swappiness for better performance
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Check current GPU utilization
echo "Current GPU utilization:"
sudo tegrastats | head -n 2

echo ""
echo "Optimization complete! Your Jetson Xavier NX is now configured for GPU-accelerated robotics."
echo "Please run 'source ~/.bashrc' or restart your terminal for changes to take effect."
echo ""
echo "To monitor GPU usage during operations, run: 'sudo tegrastats'"
echo "You should see GPU utilization (GR3D) increase during navigation and mapping."