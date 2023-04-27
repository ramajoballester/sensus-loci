import cv2
import json
import os
import numpy as np
from pypcd import pypcd
import shutil
import argparse

def load_json(file):
    # Abrir el archivo en modo de lectura
    with open(file, 'r') as file:
        # Cargar los datos del archivo en forma de un objeto Python
        datos = json.load(file)
    return datos

def create_dirs(dest_dir):
    # Create all necessary directories
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(dest_dir + '/images'):
        os.makedirs(dest_dir + '/images')
    if not os.path.exists(dest_dir + '/images/images_0'):
        os.makedirs(dest_dir + '/images/images_0')
    if not os.path.exists(dest_dir + '/labels'):
        os.makedirs(dest_dir + '/labels')
    if not os.path.exists(dest_dir + '/calibs'):
        os.makedirs(dest_dir + '/calibs')
    if not os.path.exists(dest_dir + '/points'):
        os.makedirs(dest_dir + '/points')
    if not os.path.exists(dest_dir + '/ImageSets'):
        os.makedirs(dest_dir + '/ImageSets')

def adapt_calib(src_dir, dest_dir, file_code):
    camera_data = load_json(src_dir + '/calib/camera_intrinsic/' + file_code + '.json')
    lidar_data = load_json(src_dir + '/calib/virtuallidar_to_camera/' + file_code + '.json')
    dest_file = dest_dir + file_code + '.txt'

    camera_intrinsc = camera_data['cam_K']
    lidar_rotation = lidar_data['rotation']
    lidar_translation = lidar_data['translation']

    with open(dest_file, 'w') as archivo_txt:
        archivo_txt.write(f"P0: {camera_intrinsc[0]} {camera_intrinsc[1]} {camera_intrinsc[2]} {camera_intrinsc[3]} {camera_intrinsc[4]} {camera_intrinsc[5]} {camera_intrinsc[6]} {camera_intrinsc[7]} {camera_intrinsc[8]}\n")
        archivo_txt.write(f"lidar2cam0: {lidar_rotation[0][0]} {lidar_rotation[0][1]} {lidar_rotation[0][2]} {lidar_translation[0][0]} {lidar_rotation[1][0]} {lidar_rotation[1][1]} {lidar_rotation[1][2]} {lidar_translation[1][0]} {lidar_rotation[2][0]} {lidar_rotation[2][1]} {lidar_rotation[2][2]} {lidar_translation[2][0]} 0.0 0.0 0.0 1.0\n")

def adapt_points(pcd_file, bin_file):
    pcd_data = pypcd.PointCloud.from_path(pcd_file)
    points = np.zeros([pcd_data.width, 4], dtype=np.float32)
    points[:, 0] = pcd_data.pc_data['x'].copy()
    points[:, 1] = pcd_data.pc_data['y'].copy()
    points[:, 2] = pcd_data.pc_data['z'].copy()
    points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
    with open(bin_file, 'wb') as f:
        f.write(points.tobytes())

def adapt_labels(src, archivo_txt):
    datos = load_json(src)
    with open(archivo_txt, 'w') as archivo_txt:
        for obj in datos:
            archivo_txt.write(f"{obj['3d_location']['x']} {obj['3d_location']['y']} {obj['3d_location']['z']} {obj['3d_dimensions']['l']} {obj['3d_dimensions']['w']} {obj['3d_dimensions']['h']} {obj['rotation']} {obj['type']} \n")

def transform_files(src, images_dir, points_dir, dest, file_code):
    shutil.copy(images_dir +'/' + file_code + '.jpg', dest + '/images/images_0/')
    adapt_points(points_dir+ '/' + file_code + '.pcd', dest + '/points/' + file_code + '.bin')
    adapt_calib(src, dest + '/calibs/', file_code)
    adapt_labels(src + '/label/camera/' + file_code + '.json', dest + '/labels/' + file_code + '.txt')

def process_dataset(src_dir, images_dir, points_dir, dest_dir):
    # Create all necessary directories
    create_dirs(dest_dir)

    # Get all files in the directory
    files = os.listdir(images_dir)
    for file in files:
        file_code = file.split('.')[0]
        transform_files(src_dir, images_dir, points_dir, dest_dir, file_code)
    print('Successfully converted dataset!')

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process a dataset')

    # Add the arguments
    parser.add_argument('--src', required=True, help='the source directory')
    parser.add_argument('--images', required=True, help='the directory containing images')
    parser.add_argument('--points', required=True, help='the directory containing points')
    parser.add_argument('--dest', required=True, help='the destination directory')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    process_dataset(args.src, args.images, args.points, args.dest)

## proccess_dataset('/home/javier/datasets/DAIR/single-infrastructure-side', '/home/javier/datasets/DAIR/single-infrastructure-side-image', '/home/javier/datasets/DAIR/single-infrastructure-side-velodyne', 'home/javier/sensus-loci/dataset_converter/example')