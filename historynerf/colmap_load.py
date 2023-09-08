from pathlib import Path
import shutil
from typing import Tuple

from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary, read_points3D_binary, write_cameras_binary, write_images_binary, write_points3D_binary

class ColmapLoader:
    def __init__(
        self, 
        recon_dir: Path, 
        output_dir: Path, 
        imgs_dir: Path) -> None:
        """
        Initialize the ColmapLoader.

        Args:
            recon_dir (Path): Directory containing the colmap reconstruction files.
            output_dir (Path): Directory where the output will be saved.
            imgs_dir (Path): Directory containing the images.
        """
        self.recon_dir = recon_dir
        self.output_dir = output_dir
        self.imgs_dir = imgs_dir
        self._load_colmap()

    def _load_colmap(self) -> None:
        """Load the colmap reconstruction files."""
        self.points3D = read_points3D_binary(str(self.recon_dir / "points3D.bin"))
        self.cameras = read_cameras_binary(str(self.recon_dir / "cameras.bin"))
        self.images = read_images_binary(str(self.recon_dir / "images.bin"))

    def _copy_colmap_imgs(self, folder: str) -> None:
        """Copy the images to the output folder."""
        target_folder = self.output_dir / folder
        target_folder.mkdir(parents=True, exist_ok=False)
        for sample in self.sampled_images.values():
            shutil.copy(self.recon_dir.parents[2] / folder / sample.name, target_folder / sample.name)

    def _save_colmap(self) -> None:
        """Save the sampled colmap reconstruction."""
        self.colmap_output_dir = self.output_dir / "colmap" / "sparse" / "0"
        self.colmap_output_dir.mkdir(parents=True, exist_ok=False)
        write_cameras_binary(self.cameras, str(self.colmap_output_dir / "cameras.bin"))
        write_images_binary(self.sampled_images, str(self.colmap_output_dir / "images.bin"))
        write_points3D_binary(self.points3D, str(self.colmap_output_dir / "points3D.bin"))
        
    def undersample(self) -> None:
        """Sample the colmap reconstruction based on images in imgs_dir."""
        # Get filenames of the images (.jpg) in ims_dir
        filenames = [f.name for f in self.imgs_dir.iterdir() if f.suffix == ".jpg"]
        # Check if sampling colmap is necessary
        if len(filenames) != len(self.images):
            # Get the colmap reconstruction (images) for the images in filenames
            self.sampled_images = {k: v for k, v in self.images.items() if v.name in filenames}
            for folder in ["images", "images_2", "images_4", "images_8"]:
                self._copy_colmap_imgs(folder)
            self._save_colmap()

    def update_path(self, current_colmap: Path, current_input: Path) -> Tuple[Path, Path]:
        """
        Update and return the colmap and input paths.

        Args:
            current_colmap (Path): Current colmap directory.
            current_input (Path): Current input directory.

        Returns:
            Tuple[Path, Path]: Updated colmap and input directories.
        """
        if hasattr(self, 'colmap_output_dir') and self.colmap_output_dir:
            return self.colmap_output_dir, self.output_dir / "images"
        return current_colmap, current_input

