#!/bin/bash

# Update and install dependencies
sudo apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gnupg2 \
    libopenblas-dev \
    lsb-release \
    net-tools \
    python3-pip \
    python3-dev \
    python3-rosdep2 \
    ros-noetic-realsense2-camera \
    software-properties-common \
    usbutils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ROS Noetic
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

rosdep init && rosdep update

# Check if the ROS setup line is already in .bashrc
if ! grep -Fxq "source /opt/ros/noetic/setup.bash" ~/.bashrc
then
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
fi

# Set the CUDA installer URL and the filename
CUDA_URL="http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run"
CUDA_FILENAME="cuda_11.0.2_450.51.05_linux.run"

# Check if CUDA is already installed
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA is already installed, skipping installation."
else
    # Check if the CUDA installer has already been downloaded
    if [ ! -f "$CUDA_FILENAME" ]; then
        wget "$CUDA_URL"
    else
        echo "$CUDA_FILENAME already exists, skipping download."
    fi

    # Run the CUDA installer if the file exists
    if [ -f "$CUDA_FILENAME" ]; then
        sudo sh "$CUDA_FILENAME" --toolkit --silent --override
    else
        echo "CUDA installer not found, cannot proceed with installation."
    fi
fi

# Function to remove duplicate paths
remove_duplicate_paths() {
    echo "$1" | awk -v RS=':' '!seen[$0]++' | paste -sd ':'
}

# Clean and export environment variables
export CUDA_HOME=/usr/local/cuda-11.0
export PATH=$(remove_duplicate_paths "${CUDA_HOME}/bin:${PATH}")
export LD_LIBRARY_PATH=$(remove_duplicate_paths "${CUDA_HOME}/lib64:$LD_LIBRARY_PATH")

# Check if conda is installed, install Miniconda if not
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found, installing Miniconda."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
    wget "$MINICONDA_URL" -O "$MINICONDA_SCRIPT"
    bash "$MINICONDA_SCRIPT" -b -p $HOME/miniconda
    rm "$MINICONDA_SCRIPT"
    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/bin/activate"
else
    echo "Conda is already installed."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Create and activate the Conda environment
ENV_NAME="anygrasp"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists, removing it."
    conda deactivate
    conda env remove -n "$ENV_NAME" -y
fi
echo "Creating Conda environment '$ENV_NAME'."
conda create -y -n "$ENV_NAME" python=3.8
export PYTHONNOUSERSITE=True
export PYTHON="$HOME"/miniconda/envs/"$ENV_NAME"/bin/python
source activate "$ENV_NAME"
conda install pip
export MAX_JOBS=2

# Install necessary Python packages
conda install -y openblas-devel -c anaconda
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

"$HOME"/miniconda/envs/"$ENV_NAME"/bin/pip install -U scikit-learn numpy==1.20.3 Pillow scipy tqdm graspnetAPI open3d 
"$HOME"/miniconda/envs/"$ENV_NAME"/bin/pip install -U git+https://github.com/NVIDIA/MinkowskiEngine


# Install PointNet2
cd pointnet2
python3 setup.py install
cd ..

# Latest version of numpy needed to install minkowskiEngine but oldest version needed to use anygrasp
"$HOME"/miniconda/envs/"$ENV_NAME"/bin/pip install numpy==1.23.4 rospkg roboticstoolbox-python
