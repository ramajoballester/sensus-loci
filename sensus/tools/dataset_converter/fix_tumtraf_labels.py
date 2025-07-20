import os
import argparse


def fix_label_file(file_path):
    """Fix a single label file by converting labels to lowercase and first number to int"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                # Convert label to lowercase
                parts[0] = parts[0].lower()
                # Convert first number from float to int (0.0 -> 0)
                parts[1] = str(int(float(parts[1])))

            fixed_lines.append(' '.join(parts) + '\n')

        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)

        return f'Fixed: {file_path}'

    except Exception as e:
        return f'Error fixing {file_path}: {e}'


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fix TUMTraf label files: convert labels to lowercase and first number to integer'
    )
    parser.add_argument(
        'directory', help='Directory containing .txt label files'
    )
    args = parser.parse_args()

    # Validate directory exists
    if not os.path.isdir(args.directory):
        print(f'Error: Directory "{args.directory}" does not exist')
        exit(1)

    # Find all .txt files
    txt_files = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    print(f'Found {len(txt_files)} .txt files to process')

    # Process each file
    for txt_file in txt_files:
        result = fix_label_file(txt_file)
        print(result)

    print('Processing complete!')


if __name__ == '__main__':
    main()
