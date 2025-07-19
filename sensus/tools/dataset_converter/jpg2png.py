import os
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp


def convert_image(jpg_path):
    png_path = jpg_path[:-4] + '.png'
    try:
        img = Image.open(jpg_path)
        img.save(png_path)
        return f'Converted: {jpg_path}'
    except Exception as e:
        return f'Error converting {jpg_path}: {e}'


# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert JPG files to PNG format')
parser.add_argument('directory', help='Directory to search for JPG files')
args = parser.parse_args()

# Validate directory exists
if not os.path.isdir(args.directory):
    print(f'Error: Directory "{args.directory}" does not exist')
    exit(1)

# Collect all jpg files
jpg_files = []
for root, dirs, files in os.walk(args.directory):
    for file in files:
        if file.lower().endswith('.jpg'):
            jpg_files.append(os.path.join(root, file))

print(f'Found {len(jpg_files)} JPG files to convert')

# Use all CPU cores
num_cores = mp.cpu_count()
print(f'Using {num_cores} CPU cores')

# Convert with multiprocessing and progress bar
with Pool(num_cores) as pool:
    results = list(
        tqdm(
            pool.imap(convert_image, jpg_files),
            total=len(jpg_files),
            desc='Converting JPG to PNG',
        )
    )

# Print any errors
for result in results:
    if 'Error' in result:
        print(result)
