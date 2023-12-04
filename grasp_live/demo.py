import os
import argparse
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from graspnetAPI import GraspGroup
from tracker import AnyGraspTracker
from gsnet import AnyGrasp
import rospy
from std_msgs.msg import String
import time

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--method', type=String, default="detection", help='Method to get grasping positions')
cfgs = parser.parse_args()

class RealSense2Camera:
    def __init__(self):
        rospy.init_node('Anygrasp', anonymous=False)
        
        # self.cv_bridge = CvBridge()
        self.current_image, self.current_depth = None, None
        self.current_point_cloud = None


        # Camera
        self.frame_id = "robot_k4a_bottom_depth_camera_link"
        self.height, self.width = None, None
        
        self.color_info_topic = "/camera/color/camera_info"
        self.depth_info_topic = "/camera/aligned_depth_to_color/camera_info"
        self.intrinsics_topic = "/camera/color/camera_info"
        
        self.color_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"

        # From rostopic /camera/aligned_depth_to_color/camera_info K:
        #TODO get this from rostopic
        self.camera_intrinsics = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.scale = None

        self.color_sensor_state = {'active': False, 'ready': False}
        self.depth_sensor_state = {'active': False, 'ready': False}

        # Subscribers
        self.image_sub = rospy.Subscriber(
            self.color_topic, Image, self.callback_receive_color_image, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.callback_receive_depth_image, queue_size=1)
        self.sub = rospy.Subscriber(
            self.intrinsics_topic, CameraInfo, self.callback_intrinsics, queue_size=1)
        
        # Publishers
        self.grasp_pose_pub = rospy.Publisher('/grasp_pose', String, queue_size=10)
        
        self.rate = rospy.Rate(6)
        

    def _active_sensor(self):
        self.color_sensor_state['active'] = True
        self.depth_sensor_state['active'] = True

    def callback_intrinsics(self, data):
        self.intrinsics = data
        
        self.height, self.width = self.intrinsics.height, self.intrinsics.width
        self.camera_intrinsics = np.array(self.intrinsics.K).reshape(3, 3)
        self.fx = self.camera_intrinsics[0, 0]
        self.fy = self.camera_intrinsics[1, 1]
        self.cx = self.camera_intrinsics[0, 2]
        self.cy = self.camera_intrinsics[1, 2]
        self.scale = 1/0.001
        self.sub.unregister()

    def callback_receive_color_image(self, image):
        if not self.color_sensor_state['active']:
            return

        # Get BGR image from data
        current_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        # Transfer BGR into RGB
        self.current_image = current_image[:, :, [2, 1, 0]]

        self.color_sensor_state['active'] = False
        self.color_sensor_state['ready'] = True

    def callback_receive_depth_image(self, depth):
        """ Callback. Get raw depth from data (Unit: mm). """

        if not self.depth_sensor_state['active']:
            return
        """
            Reference here:
                current_depth = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")
        """

        # Way 1: works
        if depth.encoding == '16UC1':
            channel = 1
            dtype = np.dtype('uint16')
            dtype = dtype.newbyteorder('>' if depth.is_bigendian else '<')

        # NOTE! not sure
        elif depth.encoding == '32FC1':
            channel = 1
            dtype = np.dtype('float32')
            dtype = dtype.newbyteorder('>' if depth.is_bigendian else '<')

        current_depth = np.frombuffer(depth.data, dtype=dtype).reshape(
            depth.height, depth.width, channel)

        # Way 2: works
        # if depth.encoding == '16UC1':
        #     depth.encoding = "mono16"
        #     current_depth = self.cv_bridge.imgmsg_to_cv2(depth, "mono16")

        # elif depth.encoding == '32FC1':
        #     depth.encoding = "mono16"
        #     current_depth = self.cv_bridge.imgmsg_to_cv2(depth, "mono16")


        # Way 3: works
        # current_depth = self.cv_bridge.imgmsg_to_cv2(
        #     depth, desired_encoding="passthrough")

        # Convert unit from millimeter into meter
        # current_depth = current_depth.astype(float) / 1000.
        current_depth = current_depth.astype(float)

        self.current_depth = current_depth.squeeze(axis=2) \
                             if len(current_depth.shape) >= 3 else current_depth

        self.depth_sensor_state['active'] = False
        self.depth_sensor_state['ready'] = True

    def get_rgbd_images(self):
        self._active_sensor()
        i = 0
        while True:
            if (self.color_sensor_state['ready'] and
                self.depth_sensor_state['ready']):
                color_image = self.current_image
                depth_image = self.current_depth

                self.color_sensor_state['ready'] = False
                self.depth_sensor_state['ready'] = False
                return color_image, depth_image

            rospy.sleep(0.1)
            i += 1
            print(i, end='\r')
            if i >= 30:
                print("No image")
                exit()
                
    def point_from_depth_image(self, depth, organized=True):
        """ Generate points using depth image only.\\
            Args:
                depth    (np.float): Depth image with the shape (H,W)
                organized    (bool): True for keeping the cloud in image shape (H,W,3)
            Returns:
                cloud: (np.float): points with shape (H,W,3) or (H*W,3) / point cloud
        """

        assert(depth.shape[0] == self.height and depth.shape[1] == self.width)
        xmap = np.arange(self.width)
        ymap = np.arange(self.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth
        points_x = (xmap - self.cx) * points_z / self.fx
        points_y = (ymap - self.cy) * points_z / self.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        return cloud

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    # TODO: delete this and directly use PC from ROS topic of the camera
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        points = points.reshape([-1, 3])
    return points

def get_data(camera):
    # load image
    colors, depths = camera.get_rgbd_images()
    colors = colors / 255

    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
    points = points[mask]
    colors = colors[mask]

    return points, colors

def demo():
    # intialization
    # TODO remove tracking once sure detection is the good method
    # NOTE "--checkpoint_path" must be given accordingly in json file
    if cfgs.method.data == "tracking":
        anygrasp_tracker = AnyGraspTracker(cfgs)
        anygrasp_tracker.load_net()
    elif cfgs.method.data == "detection":
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
    
    camera = RealSense2Camera()
    grasp_ids = [0]
    while camera.scale == None:
        print("Waiting for camera intrinsics", end='\r')
        camera.rate.sleep()
    print("intrinsics parameters acquired")
        
    if cfgs.debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=camera.height, width=camera.width)
    
    
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    for i in range(1000):
        # get prediction
        points, colors = get_data(camera)
        
        if cfgs.method.data == "tracking":
            target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
            if i == 0:
                # select grasps on objects to track for the 1st frame
                grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
                grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
                grasp_mask_z = ((curr_gg.translations[:,2]>0.35) & (curr_gg.translations[:,2]<0.55))
                grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][:30:6]
                target_gg = curr_gg[grasp_ids]
            else:
                grasp_ids = target_grasp_ids
            #print(i, target_grasp_ids)
            
            best_index = np.argmax(target_gg.scores)
            best_depth = target_gg.depths[best_index]
            best_height = target_gg.heights[best_index]
            best_width = target_gg.widths[best_index]
            best_trans = target_gg.translations[best_index]
            best_rot = target_gg.rotation_matrices[best_index]
            
            print(best_trans)
            camera.grasp_pose_pub.publish(str(best_trans))
            
            # visualization
            if cfgs.debug:
                trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                cloud.transform(trans_mat)
                grippers = target_gg.to_open3d_geometry_list()
                for gripper in grippers:
                    gripper.transform(trans_mat)
                vis.add_geometry(cloud)
                for gripper in grippers:
                    vis.add_geometry(gripper)
                vis.poll_events()
                vis.remove_geometry(cloud)
                for gripper in grippers:
                    vis.remove_geometry(gripper)
        


        elif cfgs.method.data == "detection":
            points = np.float32(points)
            colors = np.float32(colors)
            gg, cloud = anygrasp.get_grasp(points, colors, lims)

            if len(gg) == 0:
                print('No Grasp detected after collision detection!')

            gg = gg.nms().sort_by_score()
            target_gg = gg[0:20]
            print(target_gg.scores)
            print('grasp score:', target_gg[0].score)
            
            best_index = np.argmax(target_gg.scores)
            best_depth = target_gg.depths[best_index]
            best_height = target_gg.heights[best_index]
            best_width = target_gg.widths[best_index]
            best_trans = target_gg.translations[best_index]
            best_rot = target_gg.rotation_matrices[best_index]
            
            print(best_trans)
            camera.grasp_pose_pub.publish(str(best_trans))
            
            if cfgs.debug:
                trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                cloud.transform(trans_mat)
                grippers = target_gg.to_open3d_geometry_list()
                for gripper in grippers:
                    gripper.transform(trans_mat)
                vis.add_geometry(cloud)
                for gripper in grippers:
                    vis.add_geometry(gripper)
                vis.poll_events()
                vis.remove_geometry(cloud)
                for gripper in grippers:
                    vis.remove_geometry(gripper)

        
        camera.rate.sleep()

if __name__ == "__main__":
    demo()
