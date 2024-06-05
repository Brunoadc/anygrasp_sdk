roslaunch realsense2_camera rs_camera.launch  align_depth:=true \
filters:=pointcloud \
color_fps:=30 depth_fps:=30 \
color_height:=720 color_width:=1280 \
depth_height:=720 depth_width:=1280 \
json_file_path:="/home/bruno/Document/darko/anygrasp_sdk/grasp_robot/close_range.json" 

# color_fps:=30 depth_fps:=60 \
# color_height:=480 color_width:=640  \
# depth_height:=480 depth_width:=640 \
# color_fps:=30 depth_fps:=30 \
# color_height:=720 color_width:=1280  \
# depth_height:=720 depth_width:=1280 \





