from historynerf.pose_estimation.base import PoseEstimator
from historynerf.pose_estimation.configs import COLMAPConfig
from historynerf.pose_estimation.evaluation import compare_poses
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
            verbose = False,
        )
    
    def _match_features(self):
        # Match features between images
        if self.config.matching_method == "exhaustive":
            pycolmap.match_exhaustive(
                database_path = self.config.database_path,
                verbose = False,
                # default values {"max_ratio": 0.8, "max_distance": 0.7, "min_num_inliers": 15}
                # sift_options = {"max_ratio": 0.8, "max_distance": 0.7, "min_num_inliers": 5},
            )
        elif self.config.matching_method == "sequential":
            pycolmap.match_sequential(
                database_path = self.config.database_path,
                verbose = False,
            )
        elif self.config.matching_method == "spatial":
            pycolmap.match_spatial(
                database_path = self.config.database_path,
                verbose = False,
            )
        elif self.config.matching_method == "vocabtree":
            pycolmap.match_vocabtree(
                database_path = self.config.database_path,
                verbose = False,
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
        if self.config.image_list:
            output_dir = Path(self.config.output_dir) / f"chosen_nsamples{len(self.config.image_list)}"
            if output_dir.exists():
                output_dir = Path(self.config.output_dir) / f"chosen_nsamples{len(self.config.image_list)}_{self.config.seed}"
            output_dir.mkdir()
            with open(str(output_dir / f"undersample_list.txt"), "w") as f:
                f.write("\n".join(self.config.image_list))
            undersample_image_list = self.config.image_list
        else:    
            undersample_image_list, output_dir = undersample_images(
                image_path=self.config.image_dir,
                output_dir=self.config.output_dir, 
                rnd_seed=self.config.seed, 
                sample_size=self.config.sample_size)
        self.config.output_dir = output_dir

        self._extract_features(image_list = undersample_image_list)
        self._match_features()
        maps = self._incremental_mapping()
        # Remove the 0 folder, which is the initial reconstruction, if it exists
        if (Path(self.config.output_dir) / "0").exists():
            shutil.rmtree((Path(self.config.output_dir) / "0"))
        
        if bool(maps):
            # Write the output to the output directory, i.e. cameras, images and points3D. The output directory is changed if undersampling is used. 
            maps[0].write(self.config.output_dir)
        else:
            print("No reconstruction found")
        
    
    def evaluate_poses(self):
        return compare_poses(
            path1=self.config.gt_poses_dir, 
            path2=self.config.output_dir,
            angular_error_max_dist=self.config.angular_error,
            translation_error_max_dist=self.config.translation_error)