from nerfstudio.utils.io import load_from_json
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary, read_points3D_binary, write_cameras_binary, write_images_binary, write_points3D_binary

class ColmapLoader:
    def __init__(self, recon_dir, output_dir, imgs_dir):
        self.recon_dir = recon_dir
        self.output_dir = output_dir
        self.imgs_dir = imgs_dir
        self.load_colmap()
        self.sample_colmap()

    def load_colmap(self):
        self.points3D = read_points3D_binary(str(self.recon_dir / "points3D.bin"))
        self.cameras = read_cameras_binary(str(self.recon_dir / "cameras.bin"))
        # Camera is just one, might need to only modify the images and copy the rest
        self.images = read_images_binary(str(self.recon_dir / "images.bin"))
        breakpoint()
    
    def sample_colmap(self):
        # Get filenames of the images (.jpg) in ims_dir
        filenames = [f.name for f in self.imgs_dir.iterdir() if f.suffix == ".jpg"]
        # Get the colmap reconstruction (images) for the images in filenames
        sampled_images = {k:v for k,v in self.images.items() if v.name in filenames}
        breakpoint()

    def save_colmap(self):
        write_cameras_binary(self.cameras, str(self.output_dir / "cameras.bin"))
        write_images_binary(self.images, str(self.output_dir / "images.bin"))
        write_points3D_binary(self.points3D, str(self.output_dir / "points3D.bin"))

    


