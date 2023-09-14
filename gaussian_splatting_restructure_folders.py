import os
import shutil
import argparse


def restructure_folders(source_location, target_location):
    # Ensure the target_location exists; if not, create it
    os.makedirs(target_location, exist_ok=True)
    
    # Define the paths for source directories
    images_src = os.path.join(source_location, 'images')
    colmap_src = os.path.join(source_location, 'colmap')
    
    # Define the paths for destination directories
    input_dest = os.path.join(target_location, 'input')
    distorted_dest = os.path.join(target_location, 'distorted')
    
    # Copy the entire 'images' folder to 'input'
    if not os.path.exists(input_dest):
        shutil.copytree(images_src, input_dest)
    
    # Copy the contents of 'colmap' to 'distorted'
    for item in os.listdir(colmap_src):
        item_src_path = os.path.join(colmap_src, item)
        item_dest_path = os.path.join(distorted_dest, item)
        
        if os.path.isdir(item_src_path):
            shutil.copytree(item_src_path, item_dest_path)
        else:
            shutil.copy2(item_src_path, item_dest_path)


def main():
    parser = argparse.ArgumentParser(description="Restructure folders from source to target location.")
    parser.add_argument("source_location", help="Path to the base directory with the initial structure.")
    parser.add_argument("target_location", help="Path to the target directory where the new structure will be created.")
    args = parser.parse_args()
    
    restructure_folders(args.source_location, args.target_location)

if __name__ == "__main__":
    main()
