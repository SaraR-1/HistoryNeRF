from historynerf.pose_estimation.base import PoseEstimator
from historynerf.pose_estimation.configs import COLMAPConfig
from historynerf.pose_estimation.evaluation import compare_poses
from pathlib import Path
import pycolmap
import shutil

from historynerf.pose_estimation.utils.data_preprocessing import undersample_images, undersample_video_frames, undersample_fromlist


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
            device="cuda",
            verbose = True,
        )
    
    def _match_features(self):
        # Match features between images
        if self.config.matching_method == "exhaustive":
            pycolmap.match_exhaustive(
                database_path = self.config.database_path,
                verbose = True,
                sift_options = {"gpu_index": self.config.use_gpu},
                device="cuda",
                # default values {"max_ratio": 0.8, "max_distance": 0.7, "min_num_inliers": 15}
                # sift_options = {"max_ratio": 0.8, "max_distance": 0.7, "min_num_inliers": 5},
            )
        elif self.config.matching_method == "sequential":
            pycolmap.match_sequential(
                database_path = self.config.database_path,
                verbose = False,
                sift_options = {"gpu_index": self.config.use_gpu},
                device="cuda",
            )
        elif self.config.matching_method == "spatial":
            pycolmap.match_spatial(
                database_path = self.config.database_path,
                verbose = False,
                sift_options = {"gpu_index": self.config.use_gpu},
                device="cuda",
            )
        elif self.config.matching_method == "vocabtree":
            pycolmap.match_vocabtree(
                database_path = self.config.database_path,
                verbose = False,
                sift_options = {"gpu_index": self.config.use_gpu},
                device="cuda",
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
        # Init empty list of images to undersample and output directory
        undersample_image_list = []
        output_dir = self.config.output_dir

        if self.config.image_list:
            undersample_image_list, output_dir = undersample_fromlist(
                output_dir=self.config.output_dir,
                image_list=self.config.image_list,
                rnd_seed=self.config.seed,)
            
        elif self.config.sample_size:    
            undersample_image_list, output_dir = undersample_images(
                image_path=self.config.image_dir,
                output_dir=self.config.output_dir, 
                rnd_seed=self.config.seed, 
                sample_size=self.config.sample_size)
        
        elif self.config.video_sample_step:
            undersample_image_list, output_dir = undersample_video_frames(
                video_path=self.config.video_path,
                output_dir=self.config.output_dir,
                step_size=self.config.video_sample_step,)
            
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