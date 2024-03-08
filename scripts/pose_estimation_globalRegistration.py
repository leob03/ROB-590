import rospy
import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from IPython import embed
from copy import deepcopy
import math
"""
Set intrinsic parameters for Microsoft Azure Depth camera
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

"""
#camera intrinsic parameters 
fx = 505.103
fy = 505.2
cx = 325.034
cy = 339.758
width = 640
height = 576

class ImageProcessor:
    """
        Initializes the ImageProcessor object. Sets up ROS subscribers for RGB and depth images,
        initializes variables for storing images and point clouds, and prints a message indicating
        that the topics have been subscribed to.
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/rgb_to_depth/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/depth/image_raw", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None
        self.source = None
        self.target = None
        self.filtered_source = None
        self.timer = rospy.Timer(rospy.Duration(1/10), self.timer_callback)
        #self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size =0.1, origin = [0,0,0])
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        # self.pcd = o3d.geometry.PointCloud()
        #print("Subscribed to topics")
    
    def timer_callback(self, event):
        if self.color_image is not None and self.depth_image is not None:
            self.process_images()

            # Visualization in loop (not working fn)
            # if self.filtered_source.is_empty():
            #     self.vis.add_geometry(self.filtered_source)
            # else:
            #     self.vis.update_geometry(self.filtered_source)

            # if self.target.is_empty():
            #     self.vis.add_geometry(self.target)
            # else:
            #     self.vis.update_geometry(self.target)
            
            # # if not self.vis.has_geometry(self.coordinate_frame):
            # #     self.vis.add_geometry(self.coordinate_frame)
            # # else:
            # #     self.vis.update_geometry(self.coordinate_frame)
            
            # self.vis.poll_events()
            # self.vis.update_renderer()

 
    # def run(self):
    #     while not rospy.is_shutdown():
    #         # if self.vis.get_viewer_flags().is_set('quit'):
    #         #     break
    #         rospy.sleep(0.1)
    #     self.vis.destroy_window()
            

    def color_callback(self, data):
        # TODO: revisit these callbacks. 
        # In general they should only update class variables (self.color_image = ...)
        # and not do processing.
        """
        Callback function for the RGB image subscriber. Converts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #print("got color img")
            #cv2.imwrite('/home/bartonlab-user/workspace/src/graspmixer_demo/scripts/rgb_image.png', self.color_image)
            # self.process_images()
            # self.color_sub.unregister()
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        """
        Callback function for the depth image subscriber. Converts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            #print("got depth img")
            # cv2.imwrite('/home/bartonlab-user/workspace/src/graspmixer_demo/scripts/depth_image.jpeg', self.depth_image)

            # self.process_images()
            # self.depth_sub.unregister()
        except CvBridgeError as e:
            print(e)

    def process_images(self):
        """
        Processes the RGB and depth images to perform object segmentation, point cloud creation, filtering,
        registration, and visualization.
        """
        if self.color_image is not None and self.depth_image is not None:
            print("inside processing")
            
            
            segmented_mask = self.get_segmented_object(self.color_image)
            segmented_depth_image = cv2.bitwise_and(self.depth_image, self.depth_image, mask=segmented_mask)
            
            color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(segmented_depth_image.astype(np.float32))
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
            camera_instrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            self.source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_instrinsics)
   
            self.filtered_source = self.filter_source_point_cloud(self.source)
            

            print("Number of points in filtered source : ", self.filtered_source) 
            
            self.target = self.load_target_mesh()

            #Compute and Align to scale
            #self.filtered_source, self.target = self.compute_and_align_to_scale(self.filtered_source, self.target) #rn the scale factor has been set to 1 as I'm downscaling in load_target_mesh
            #aligned_pcd = self.align_major_axis_to_z(self.filtered_source)
            

            #camera coordinates
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin = [0,0,0])
            o3d.visualization.draw_geometries([self.filtered_source, self.target, self.coordinate_frame])
            


            #source body frame coordinates
            self.source_centroid = np.mean(np.asarray(self.filtered_source.points), axis=0)
            flip_transform = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])

           
            flipped_source_centroid = flip_transform @ self.source_centroid 

            self.source_coordinate_frame =  o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin =  self.source_centroid )
            print(f"Location of source : x = {self.source_centroid[0]}, y = {self.source_centroid[1]}, z = {self.source_centroid[2]}")
             
            #target body frame coordinates
            self.target_centroid = np.mean(np.asarray(self.target.points), axis=0)
            self.target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.target_centroid)
            print(f"Location of target : x = {self.target_centroid[0]}, y = {self.target_centroid[1]}, z = {self.target_centroid[2]}")
 
            #o3d.visualization.draw_geometries([self.filtered_source, self.target, self.coordinate_frame, self.source_coordinate_frame, self.target_coordinate_frame]) #visualize both source and target in the coordinate frame
          
            voxel_size = 0.015 #1.5cm
            self.filtered_source, self.target, self.filtered_source_down, self.target_down, self.filtered_source_fpfh, self.target_fpfh = self.prepare_dataset(voxel_size, self.filtered_source, self.target,self.source_centroid)

            #execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            result_ransac = self.execute_global_registration(self.filtered_source_down,  self.target_down, self.filtered_source_fpfh, self.target_fpfh, voxel_size)
            print(" Transformation obtained from Global Registration :\n ", result_ransac.transformation)
            #self.target_centroid = result_ransac.transformation @self.target_centroid
            #self.target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.target_centroid[:3])
            #visualize rough alignment
            #self.draw_registration_result(self.filtered_source_down, self.target_down, result_ransac.transformation) #visualize global registration result

            result_icp = self.refine_registration(self.filtered_source, self.target, self.filtered_source_fpfh, self.target_fpfh, result_ransac.transformation, voxel_size)
            #self.target_centroid = result_icp.transformation @ self.target_centroid
            #self.target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.target_centroid[:3])
            print("\n\n")
            print("*********************FINAL TRANSFORMATION***************************")
            print(result_icp) #prints metrics
            print(result_icp.transformation)
             
            roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(result_icp.transformation)
            print(f"Roll = {math.degrees(roll)}, Pitch = {math.degrees(pitch)}, Yaw = {math.degrees(yaw)}, x = {x}, y = {y}, z = {z}")

            # print("Inverse of ICP result")
            # self.T_inv = np.linalg.inv(result_icp.transformation)
            # print(self.T_inv)

            #roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(self.T_inv)
            #print(f"Roll = {math.degrees(roll)}, Pitch = {math.degrees(pitch)}, Yaw = {math.degrees(yaw)}, x = {x}, y = {y}, z = {z}")
        
            #self.draw_registration_result(self.filtered_source, self.target, result_ransac.transformation)  #visualize global registration
            self.draw_registration_result(self.filtered_source, self.target, result_icp.transformation)  #visualize icp result
            #self.draw_registration_result(self.filtered_source, self.target, self.T_inv ) #visualize inverted transformation

            #embed()
            print("==== Done ====")

    def align_major_axis_to_z(self, pcd):
        points = np.asarray(pcd.points)
        mean = np.mean(points, axis=0)
        cov_matrix = np.cov(points - mean, rowvar=False)

        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        major_axis = eig_vec[:, np.argmax(eig_val)]

        z_axis = np.array([0,0,1])
        rotation_axis = np.cross(major_axis, z_axis)
        rotation_angle = np.arccos(np.dot(major_axis, z_axis))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis*rotation_angle)

        pcd.rotate(rotation_matrix, center=mean)
        return pcd


    def extract_rpy_xyz(self, transformation):
        R = transformation[:3,:3]
        T = transformation[:3,3]
        #embed()
        yaw = math.atan2(R[1,0],R[0,0])
        pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = math.atan2(R[2,1], R[2,2])
        x,y,z = T
        return roll,pitch,yaw, x, y, z
            

    def refine_registration(self, source, target, source_fpfh, target_fpfh, initial_transformation, voxel_size):
        """
        Refines the registration between the source and target point clouds using the Iterative Closest Point (ICP) algorithm.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the target point cloud.
        initial_transformation (np.ndarray): The initial transformation matrix.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
        """
        distance_threshold = voxel_size*1
        print("Applying ICP")
        print(" distance threshold = ", distance_threshold)

        # tform_ransac = initial_transformation
        # initial_transformation = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp(
             target, source, 0.05, np.linalg.inv(initial_transformation),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(  max_iteration = 10000)
        )

     
        return reg_p2p

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        """
        Performs global registration between downsampled source and target point clouds using RANSAC and feature matching.

        Parameters:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the downsampled source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the downsampled target point cloud.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the global registration.
        """

        print("================INSIDE GLOBAL REGISTRATION========================")
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)

        best_result = None
        best_fitness = 0
        num_initializations = 5

        for i in range(num_initializations):
            initial_transformation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, 3))
            initial_transformation = np.vstack((initial_transformation, [0,0,0]))
            initial_transformation = np.column_stack((initial_transformation, [0,0,0,1]))
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                5, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
            
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_result = result
        return result

    def prepare_dataset(self, voxel_size, source, target, centroid):
        """
        Prepares the dataset for registration by transforming the source point cloud, downsampling both point clouds,
        and computing FPFH features.

        Parameters:
        voxel_size (float): The voxel size used for downsampling.
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        centroid (np.ndarray): The centroid of the source point cloud.

        Returns:
        tuple: A tuple containing the source point cloud, target point cloud, downsampled source point cloud,
               downsampled target point cloud, FPFH feature of the downsampled source point cloud, and FPFH feature
               of the downsampled target point cloud.
        """

        print("=============INSIDE PREPARE DATASET===================")
        print(":: Load two point clouds and disturb initial pose.")
      
        source.transform( np.identity(4))
        #self.draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        Preprocesses a point cloud by downsampling, estimating normals, and computing FPFH features.

        Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to preprocess.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        tuple: A tuple containing the downsampled point cloud and its FPFH feature.
        """
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
       
    def compute_and_align_to_scale(self, filtered_source, target):
        """
        Computes the scale factor between the source and target point clouds and aligns them accordingly
        (currently unused as the scale factor is set to 1).

        Parameters:
        filtered_source (o3d.geometry.PointCloud): The filtered source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.

        Returns:
        tuple: A tuple containing the scaled source point cloud and the target point cloud.
        """
        aabb_source = filtered_source.get_axis_aligned_bounding_box()
        aabb_extent_source = aabb_source.get_extent()
        print("AABB source : ", aabb_extent_source)

        aabb_target = target.get_axis_aligned_bounding_box()
        aabb_extent_target = aabb_target.get_extent() 
        print("AABB target : ", aabb_extent_target)
        #print("source x = {}, y = {}, z = {}".format(aabb_extent_source[0], aabb_extent_source[1], aabb_extent_source[2]))
        #scale_factor = aabb_extent_target[0]/aabb_extent_source[0]  # target_  / source_
        scale_factor = 1

        scaled_source_pcd = filtered_source.scale(scale_factor, center = filtered_source.get_center())
        filtered_source = scaled_source_pcd
        aabb_after_scale = filtered_source.get_axis_aligned_bounding_box()
        aabb_after_scaling = aabb_after_scale.get_extent()
        print("Source after scaling  : ", aabb_after_scaling)
        return filtered_source, target

    def get_segmented_object(self, rgb_image):
        """
        Segments the object of interest from the RGB image using HSV color space and returns a binary mask.

        Parameters:
        rgb_image (np.ndarray): The RGB image to segment.

        Returns:
        np.ndarray: The binary mask of the segmented object.
        """
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0,0,0])   #values to segment black color object (object 6)
        upper_bound = np.array([179,255,91]) 
        # lower_bound = np.array([92,76,35])   #values to segment blue color object (object 6)     
        # upper_bound = np.array([111,255,255]) 
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(rgb_image, rgb_image, mask = mask)

        

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)


        x1,y1, x2, y2 =  180, 48, 546, 521  #can move the object, mask is for the whole 'workspace'
        mask_focused = np.zeros(contour_mask.shape[:2], dtype = np.uint8)
        mask_focused[y1:y2, x1:x2] = 255
        
        masked_image = cv2.bitwise_and(contour_mask, contour_mask, mask = mask_focused)
        
        return masked_image

    def filter_source_point_cloud(self, source):
        """
        Filters the source point cloud by removing points that are too close or too far from the camera.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud to be filtered.

        Returns:
        o3d.geometry.PointCloud: The filtered source point cloud.
        """
         
        min_distance = 0.0 # Adjust this value as needed
        max_distance = 0.40  #0.35 for obj 6
        # Filter out points that are too close to the camera origin and beyond max_distance
        filtered_points = []
        for point in np.asarray(source.points):
            distance = np.linalg.norm(point)
            if min_distance < distance < max_distance:
                filtered_points.append(point)

        
        filtered_source = o3d.geometry.PointCloud()
        filtered_source.points = o3d.utility.Vector3dVector(filtered_points)
        
        # print("********************** Visualizing filtered pcd ********************")
        # o3d.visualization.draw_geometries([self.filtered_source])
        print("In filtering")
        print(filtered_source.points)
        return filtered_source

    def load_target_mesh(self):
        """
        Loads the target mesh from an STL file, scales it, samples points from it, and returns the resulting point cloud.

        Returns:
        o3d.geometry.PointCloud: The point cloud sampled from the target mesh.
        """
        #stl_file = '/home/bartonlab-user/workspace/src/graspmixer_demo/large_obj_06.stl'
        stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_04.stl'
        mesh = o3d.io.read_triangle_mesh(stl_file)
        # mesh_scaled_centered = deepcopy(mesh).scale(0.001, center=mesh.get_center()) #because source is in mm, so scaled down to 0.001 from 1
        mesh_scaled = deepcopy(mesh).scale(0.001, center = 0*mesh.get_center()) 
        mesh = mesh_scaled
        # align it to the camera origin

        self.target = mesh.sample_points_uniformly(number_of_points=len(self.filtered_source.points))
 
        return self.target

    def perform_icp_registration(self, filtered_source, target):
        threshold =10
        # initial_transformation = np.asarray([
        #     [0.862, 0.11, -0.507, 0.5], 
        #     [-0.139, 0.967, -0.215, 0.7],
        #     [0.487, 0.255, 0.835, -1.4],
        #     [0, 0, 0, 1]
        # ])
        initial_transformation = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]])
        #self.draw_registration_result(self.source, self.target, initial_transformation)
        print("************************************************")
        print("Applying point to point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            filtered_source, target,  threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria( relative_rmse = 1e-6, relative_fitness = 1e-6, max_iteration = 100)
        )
         

        print(reg_p2p)
        print("Point to Point transformation is:")
        
        print(reg_p2p.transformation)
        self.draw_registration_result(filtered_source, target, reg_p2p.transformation)
        o3d.visualization.draw_geometries([filtered_source, target, self.coordinate_frame])
        print("==========================================")


        # print("Applying point to plane ICP")
        # reg_p2l = o3d.pipelines.registration.registration_icp(
        #     filtered_source, target, threshold, initial_transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        # )
        # print(reg_p2l)
        # print("Point to Plane transformation is:")
        # print(reg_p2l.transformation)
        # self.draw_registration_result(filtered_source, target, reg_p2l.transformation)




    def draw_registration_result(self, filtered_source, target, transformation):
        """
        Visualizes the registration result by applying the transformation to the source or target point cloud
        and displaying them alongside a coordinate frame.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The transformation matrix to be applied.
        """
        source_temp = copy.deepcopy(filtered_source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        # # Bring the depth towards the model 
        # source_temp_tf = deepcopy(source_temp).transform(transformation)
        # o3d.visualization.draw_geometries([source_temp_tf, target_temp, self.coordinate_frame])

        # Bring the model towards the depth
        tf_inv = np.linalg.inv(transformation)
        target_temp_tf = deepcopy(target).transform(tf_inv)

        target_tf = target.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_tf, self.coordinate_frame])


if __name__ == "__main__":
    rospy.init_node('image_processor', anonymous=True)
    processor = ImageProcessor()
    #processor.run()
    try:
        rospy.spin()
    except KeyboardInterrupter:
        print("Shutting down")