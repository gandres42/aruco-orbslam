## Installation
The included devcontainer does the following:
1. Install dependencies
```
sudo apt install -y libgl1 ros-lyrical-cv-bridge libboost-all-dev libcppunit-dev libepoxy-dev python3-scipy python3-open3d
```
2. Install Pangolin from https://github.com/stevenlovegrove/Pangolin

3. Install ORB-SLAM3 from https://github.com/gandres42/ORB_SLAM3.git

4. Install ORB-SLAM3 PythonBindings from https://github.com/gandres42/ORB_SLAM-PythonBindings.git

5. Add the package to your workspace and build away
```
mkdir -p aruco_ws/src && cd aruco_ws/src
git clone https://github.com/gandres42/aruco-orbslam.git
cd ../..
colcon build
source install/setup.bash
```

## Setup
### Camera Configuration
Set the input video topic using `Topics.video`, by default this is `/BlueROV2/video`.

### Aruco Configuration
Update `Aruco.dictionary` to the correct dictionary and `Aruco.tags` to include the sizes of tags by their IDS.  The current configuration is set up for the Hinsdale 2025 dataset.  These are only used for scale recovery.

### Anchor Configuration
A single tag is used as an anchor to place SLAM outputs in a set frame.  Update `Anchor` settings to provide the real position and rotation of a single tag.  The current configuration is set up for the Hinsdale 2025 dataset.

### SLAM Configuration
Toggle ORB-SLAM3's Pangolin visualization using `Viewer.UseViewer`

## Usage
Start the node by running `ros2 run aruco_orbslam slam`.  3D map points, poses, and paths are published according to `Topics.*` in config.yaml.  By default, these are:

`/aruco/path`: Path

`/aruco/pose`: PoseStamped

`/aruco/pointcloud`: PointCloud2