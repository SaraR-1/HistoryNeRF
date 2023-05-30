from historynerf.pose_estimation.base import PoseEstimator
from historynerf.pose_estimation.configs import COLMAPConfig
from pathlib import Path
import pycolmap
import random
import shutil

def undersample_images(
        image_path: Path, 
        output_dir: Path,
        rnd_seed: int, 
        sample_size: int):
    '''
    Unsersample the images in the image list. For sanity check, it saves the new image list in a txt file
    '''
    if sample_size:
        image_list = [str(i.name) for i in Path(image_path).iterdir()]
        random.Random(rnd_seed).shuffle(image_list)
        undersample_list = image_list[:sample_size]
        output_dir = Path(output_dir) / f"seed{rnd_seed}_nsamples{sample_size}"
        output_dir.mkdir()

        with open(str(output_dir / f"undersample_list.txt"), "w") as f:
            f.write("\n".join(undersample_list))
        return undersample_list, output_dir
    return [], output_dir

class COLMAPPoseEstimator(PoseEstimator):
    def __init__(
        self, 
        config: COLMAPConfig
    ):
        super().__init__(config)

    def _extract_features(self, image_list):
        # Extract SIFT features
        pycolmap.extract_features(
            database_path = self.config.database_path, 
            image_path = self.config.image_dir, 
            camera_model = self.config.camera_model,
            image_list = image_list,
            sift_options = {"gpu_index": self.config.use_gpu},
        )
    
    def _match_features(self):
        # Match features between images
        if self.config.matching_method == "exhaustive":
            pycolmap.match_exhaustive(
                database_path = self.config.database_path,
            )
        elif self.config.matching_method == "sequential":
            pycolmap.match_sequential(
                database_path = self.config.database_path,
            )
        elif self.config.matching_method == "spatial":
            pycolmap.match_spatial(
                database_path = self.config.database_path,
            )
        elif self.config.matching_method == "vocabtree":
            pycolmap.match_vocabtree(
                database_path = self.config.database_path,
            )
        else:
            raise ValueError(f"Matching method {self.config.matching_method} not supported")
        
    def _incremental_mapping(self):
        # Perform incremental mapping (only mapping available in COLMAP python API)
        return pycolmap.incremental_mapping(
            database_path = self.config.database_path, 
            image_path = self.config.image_dir, 
            output_path = self.config.output_dir, )

    def estimate_poses(self):
        undersample_image_list, output_dir = undersample_images(
            image_path=self.config.image_dir,
            output_dir=self.config.output_dir, 
            rnd_seed=self.config.seed, 
            sample_size=self.config.sample_size)
        self._extract_features(image_list = undersample_image_list)
        self._match_features()
        maps = self._incremental_mapping()
        shutil.rmtree((Path(self.config.output_dir) / "0"))
        # Write the output to the output directory, i.e. cameras, images and points3D. The output directory is changed if undersampling is used. 
        maps[0].write(output_dir)

        return output_dir