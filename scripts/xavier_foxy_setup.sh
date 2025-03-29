#!/bin/bash
# Comprehensive Xavier NX Setup for ROS2 Foxy with GPU Acceleration

set -e  # Exit immediately if a command exits with a non-zero status

# Logging function
log() {
    echo "[XAVIER SETUP] $1"
}

# Ensure script is run with sudo
if [[ $EUID -ne 0 ]]; then
   log "This script must be run with sudo" 
   exit 1
fi

# 1. System Performance Configuration
log "Configuring maximum performance mode..."
nvpmodel -m 0  # MAXN mode
jetson_clocks   # Max clock speeds

# CPU Performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu"
done

# 2. CUDA and GPU Dependencies
log "Installing CUDA and GPU dependencies..."
apt update
apt install -y \
    cuda-toolkit-11-4 \
    libcudnn8 \
    libcudnn8-dev \
    libopencv-dev \
    python3-opencv \
    python3-numpy

# 3. GPU Monitoring Tools
log "Installing GPU monitoring tools..."
apt install -y \
    nvidia-jetson-gpu-status \
    gpu-tools

# Attempt snap install for nvtop
if ! command -v nvtop &> /dev/null; then
    snap install nvtop || {
        log "Warning: nvtop snap installation failed. Attempting source build..."
        apt install -y cmake libncurses5-dev libncursesw5-dev git
        git clone https://github.com/Syllo/nvtop.git /tmp/nvtop
        cd /tmp/nvtop
        mkdir build && cd build
        cmake ..
        make
        make install
    }
fi

# 4. ROS2 Foxy and Dependencies
log "Ensuring ROS2 Foxy dependencies are installed..."
apt install -y \
    ros-foxy-desktop \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-rosinstall

# Initialize rosdep
rosdep init
rosdep update

# 5. CUDA Environment Configuration
log "Configuring CUDA environment..."
CUDA_CONFIG="/etc/profile.d/cuda.sh"
cat > "$CUDA_CONFIG" << 'EOF'
# CUDA Environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OPENCV_DNN_CUDA=ON
EOF

# Verify OpenCV CUDA Support
python3 -c "
import cv2
cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
print(f'OpenCV CUDA Devices: {cuda_devices}')
if cuda_devices > 0:
    print('GPU acceleration is available.')
else:
    print('WARNING: No CUDA devices detected!')
"

# 6. ROS2 Performance Tuning
log "Configuring ROS2 performance settings..."
mkdir -p /etc/ros/rosdep
cat > /etc/ros/rosdep/50-cyclonedds.xml << 'EOF'
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
  </Domain>
</CycloneDDS>
EOF

log "Xavier NX setup complete! Please reboot your system."
exit 0