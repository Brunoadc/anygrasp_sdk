# This folder contains the scripts to run anygrasp with the franka panda robot (DARKO project demo)
## Installation 
Due to [change of machine feature id for license identification](https://github.com/graspnet/anygrasp_sdk/issues/47) it is not recommended to use docker. 

1. Use the installation script to set up a conda environment with the necessary libraries. The script also installs ROS noetic if not already installed on your PC. 
```bash
    cd ..
    source install.sh
    cd grasp_robot
```
Note that this script will install CUDA Toolkit 11.0, the version is known to work with anygrasp, torch is compiled with the same CUDA version (1.7.1 with CUDA 11.0). The script won't change your GPU drivers neither it's CUDA Version.

2. Follow the instructions bellow to set gsnet, libcxx, your license and the weights correctly

## Instruction
1. Copy `gsnet.*.so` and `lib_cxx.*.so` to this folder according to your Python version (Python>=3.6,<=3.9 is supported). For example, if you use Python 3.6, you can do as follows:
```bash
    cp gsnet_versions/gsnet.cpython-36m-x86_64-linux-gnu.so gsnet.so
    cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-36m-x86_64-linux-gnu.so lib_cxx.so
```

2. Unzip your license and put the folder here as `license`. Refer to [license_registration/README.md](../license_registration/README.md) if you have not applied for license.

3. Put model weights under ``log/``.

## Execution
Start the camera node 
```bash
    bash camera.sh
```

Run the code like `demo_robot.py`, `demo.py` or any desired applications that uses `gsnet.so`.
```bash
    python3 demo.py
```
`demo.py` will plot the resulting graps directly from the camera topics.
Note that in order to use demo_robot.py (script that will publish the joint configuration for grasping) you will need the calibration matrices X1 and Y2 from the [hand eye calibration method](https://github.com/epfl-lasa/hand_eye_calibration)
