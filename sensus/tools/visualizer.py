import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
from open3d.web_visualizer import draw
import cv2
import numpy as np

from data_processor import DataProcessor

class LidarVisualizer:
    def __init__(self, bin_path, img_path):
        """
        Initialize the LidarVisualizer class

        Parameters
        ----------
        bin_path : str
            Path to the .bin file.
        img_path : str
            Path to the image file.
        """

        self.bin_path = bin_path
        self.img_path = img_path
        self.load_point_cloud()

    def load_point_cloud(self):
        with open(self.bin_path, 'rb') as f:
            points = np.fromfile(f, dtype=np.float32, count=-1).reshape([-1, 4])

        self.pcd_bin = o3d.geometry.PointCloud()
        self.pcd_bin.points = o3d.utility.Vector3dVector(points[:, :3])

    def load_labels(self, labels_path):
        self.labels = DataProcessor(labels_path, None).process_label_file()

    def load_calib(self, calib_path):
        self.calib = DataProcessor(None, calib_path).process_calib_file()

    def load_results(self, results):
        self.results = results

    def get_lidar_in_cam_T(self):
        Tr_velo_to_cam = np.array(self.calib['Tr_velo_to_cam'])
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        R = Tr_velo_to_cam[0:3, 0:3]
        T = Tr_velo_to_cam[0:3, 3]
        lidar_in_cam_T = np.eye(4)
        lidar_in_cam_T[0:3, 0:3] = R.T
        lidar_in_cam_T[0:3, 3] = -R.T @ T
        return lidar_in_cam_T
    
    def create_box(self, dimensions):
        box = o3d.geometry.TriangleMesh.create_box(width=dimensions[2], height=dimensions[1], depth=dimensions[0])
        box.paint_uniform_color([1.0, 0.0, 0.0])
        return box
    
    def convert_position_to_lidar(self, position):
        lidar_in_cam_T = self.get_lidar_in_cam_T()
        pos_in_cam = np.array(position + [1])
        pos_in_lidar = lidar_in_cam_T @ pos_in_cam
        return pos_in_lidar
    
    def translate_box(self, box, position, dimensions):
        box.translate([position[0], position[1], position[2]])
        box.translate([-dimensions[2]/2, -dimensions[1]/2, 0])
        return box
    
    def rotate_box(self, box, rotation_y):
        center = box.get_center()
        # rotation = box.get_rotation_matrix_from_xyz((0, 0, -rotation_y - np.pi/2))
        rotation = box.get_rotation_matrix_from_xyz((0, 0, +rotation_y))
        box.rotate(rotation, center=center)
        return box
    
    def generate_bbox_from_label(self, label):
        """
        Generate bbox from label dict.

        Parameters
        ----------
        label : dict
            label dictionary.
        """
        dimensions = label['dimensions']
        position = label['location']
        rotation_y = label['rotation_y']
        
        box = self.create_box(dimensions)
        position = self.convert_position_to_lidar(position)
        box = self.translate_box(box, position, dimensions)
        box = self.rotate_box(box, rotation_y)
        
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 0], [2, 3], [3, 1], [4, 5], [4, 6], [6, 7], [7, 5], [0, 4], [1, 5], [2, 6], [3, 7]]))
        lines.paint_uniform_color([1, 0, 0])
        
        return lines
    
    def generate_bbox_from_result(self, result):
        """
        Generate bbox from results array.

        Parameters
        ----------
        result : array
            results array [position, dimensions, rotation_y]
        """
        dimensions = result[3:6]
        dimensions = [dimensions[2], dimensions[1], dimensions[0]]
        position = result[0:3]
        rotation_y = result[6]

        box = self.create_box(dimensions)
        box = self.translate_box(box, position, dimensions)
        box = self.rotate_box(box, rotation_y)
        
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 0], [2, 3], [3, 1], [4, 5], [4, 6], [6, 7], [7, 5], [0, 4], [1, 5], [2, 6], [3, 7]]))
        lines.paint_uniform_color([1, 0, 0])
        
        return lines
    
    def generate_img(self, label):
        """
        Load and show img with label rectangles.

        Parameters
        ----------
        label: dict
            label dictionary
        """
        img = mpimg.imread(self.img_path)
        rect = plt.Rectangle((label['bbox'][0], label['bbox'][1]), label['bbox'][2] - label['bbox'][0], label['bbox'][3] - label['bbox'][1], linewidth=1, edgecolor='r', facecolor='none')
        fig, ax = plt.subplots(1)
        ax.add_patch(rect)
        plt.imshow(img)

    def draw_cars_from_labels(self, labels, calib, num_cars):
        """
        Draws num_cars from label file.

        Parameters
        ----------
        labels: str
            path to labels file
        calib: str
            path to calibration file
        num_cars: int
            number of cars to draw
        """
        self.load_calib(calib)
        self.load_labels(labels)

        cars = []
        for label in self.labels:
            if label['type'] == 'Car':
                cars.append(label)
            if len(cars) == num_cars:
                break

        lines = []
        for car in cars:
            lines.append(self.generate_bbox_from_label(car))
            self.generate_img(car)

        # Draw bbox into point cloud
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)

    def draw_one_car_from_label(self, label, calib):
        lines = []
        lines.append(self.generate_bbox(label, calib))
        self.generate_img(label)

        # Draw bbox into point cloud
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)

    def draw_cars_from_results(self, results, calib, num_cars):
        """
        Draws num_cars from inference results.

        Parameters
        ----------
        results: array
            mmdet inference results
        calib: str
            path to calibration file
        num_cars: int
            number of cars to draw
        """
        self.load_calib(calib)
        results = results.pred_instances_3d.bboxes_3d.tensor.to('cpu').detach().numpy()
        self.load_results(results)
        cars = []
        for result in self.results:
            cars.append(result)
            if len(cars) == num_cars:
                break
        
        lines = []
        for car in cars:
            lines.append(self.generate_bbox_from_result(car))
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)




class ImageVisualizer:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path)

    def draw_3d_bbox(self, bbox_dim, bbox_location, rotation, intrinsic_matrix, pitch):
        """
        Draw a 3D bounding box on the image.

        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the bounding box.
        bbox : list
            The 3D bounding box coordinates, it should be a list like [x, y, z, height, width, length], 
            where (x,y,z) is the center of the bbox, and (height,width,length) are the dimensions in each direction.
        rotation : float
            The rotation of the bounding box around the Y-axis.
        intrinsic_matrix : np.ndarray
            The camera intrinsic matrix.
        """

        # Extract the bbox coordinates and rotation angle
        width, height, length = bbox_dim
        print(bbox_dim)
        width, length, height = bbox_dim
        # heigh es anchura
        # width es altura
        # length es profundidad


        x, y, z = bbox_location
        r = - rotation - np.pi/2
        print(x, y, z, r)

        # Create an array to represent the bbox corners
        corners = np.array([
            [-length/2, -width/2, -height/2],
            [+length/2, -width/2, -height/2],
            [-length/2, +width/2, -height/2],
            [+length/2, +width/2, -height/2],
            [-length/2, -width/2, +height/2],
            [+length/2, -width/2, +height/2],
            [-length/2, +width/2, +height/2],
            [+length/2, +width/2, +height/2],
        ]).T

        # Apply rotation to the corners
        Ry = np.array([
            [np.cos(r), 0, np.sin(r)],
            [0, 1, 0],
            [-np.sin(r), 0, np.cos(r)],
        ])
        corners = Ry @ corners

        # Apply pitch correction to the corners
        pitch = -pitch + np.pi
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-pitch), -np.sin(-pitch)],
            [0, np.sin(-pitch), np.cos(-pitch)],
        ])
        corners = Rx @ corners

        # Translate the corners to the bbox center
        # corners += np.array([[x], [y-width/2], [z+length/2]])
        corners += np.array([[x], [y-width/2], [z+length/2]])

        # Convert the 3D bbox corners to 2D
        bbox_2d = np.dot(intrinsic_matrix, corners)
        bbox_2d = bbox_2d[:2] / bbox_2d[2]

        # Draw the bbox
        for start, end in [
            [0, 1], [2, 3], [4, 5], [6, 7], # connections in the base
            [0, 2], [1, 3], [4, 6], [5, 7], # connections in the top
            [0, 4], [1, 5], [2, 6], [3, 7]  # connections between base and top
        ]:
            start_point = tuple(np.round(bbox_2d[:, start]).astype(np.int32))
            end_point = tuple(np.round(bbox_2d[:, end]).astype(np.int32))
            cv2.line(self.image, start_point, end_point, color=(255, 0, 0), thickness=1)

        return self.image
    
    def draw_monodetection_labels(self, labels, calib, num_cars, pitch):
        cars = 0
        for label in labels:
            if label['type'] == 'Car':
                bbbox_dim = label['dimensions']
                bbox_location = label['location']
                rotation = label['rotation_y']

                intrinsic_matrix = np.array(calib['P2']).reshape(3, 4)
                intrinsic_matrix = intrinsic_matrix[:3, :3]

                # Draw the 3D bounding box
                self.image = self.draw_3d_bbox(bbbox_dim, bbox_location, rotation, intrinsic_matrix, pitch)
                cars += 1
            if cars >=num_cars:
                break
        
        # Save the image
        cv2.imwrite('viz_img.png', self.image)