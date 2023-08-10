from pathlib import Path
import shutil

from nerfstudio.utils.io import load_from_json
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary, read_points3D_binary, write_cameras_binary, write_images_binary, write_points3D_binary


class ColmapLoader:
    def __init__(self, recon_dir, output_dir, imgs_dir):
        self.recon_dir = recon_dir
        self.output_dir = output_dir
        self.imgs_dir = imgs_dir

        self._load_colmap()

    def _load_colmap(self):
        self.points3D = read_points3D_binary(str(self.recon_dir / "points3D.bin"))
        self.cameras = read_cameras_binary(str(self.recon_dir / "cameras.bin"))
        # Camera is just one, might need to only modify the images and copy the rest
        self.images = read_images_binary(str(self.recon_dir / "images.bin"))
    
    def sample_colmap(self):
        # Get filenames of the images (.jpg) in ims_dir
        filenames = [f.name for f in self.imgs_dir.iterdir() if f.suffix == ".jpg"]

        # Check if sampling colmap is necessary
        if len(filenames) != len(self.images):
            # Get the colmap reconstruction (images) for the images in filenames
            self.sampled_images = {k:v for k,v in self.images.items() if v.name in filenames}

            for folder in ["images", "images_2", "images_4", "images_8"]:
                self.copy_colmap_imgs(folder)
            
            self.save_colmap()
        else:
            self.colmap_output_dir = None

    def copy_colmap_imgs(self, folder):
        # Create the subfolders
        (self.output_dir / folder).mkdir(parents=True, exist_ok=False)

        # Copy the images
        for sample in self.sampled_images.values():
            shutil.copy(self.recon_dir.parents[2] / folder / sample.name, self.output_dir / folder / sample.name)

    def save_colmap(self):
        # Create the subfolders
        self.colmap_output_dir = self.output_dir / "colmap" / "sparse" / "0"
        (self.colmap_output_dir).mkdir(parents=True, exist_ok=False)

        write_cameras_binary(self.cameras, str(self.colmap_output_dir / "cameras.bin"))
        write_images_binary(self.sampled_images, str(self.colmap_output_dir / "images.bin"))
        write_points3D_binary(self.points3D, str(self.colmap_output_dir / "points3D.bin"))

    def update_path(self, current_colmap, current_input):
        if self.colmap_output_dir:
            return self.colmap_output_dir, self.output_dir / "images"
        return current_colmap, current_input



    


