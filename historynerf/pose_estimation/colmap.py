from historynerf.pose_estimation.base import PoseEstimator
from historynerf.pose_estimation.configs import COLMAPConfig
import pycolmap

class COLMAPPoseEstimator(PoseEstimator):
    def __init__(
        self, 
        config: COLMAPConfig
    ):
        super().__init__(config)

    def _extract_features(self):
        # Extract SIFT features
        pycolmap.extract_features(
            database_path = self.config.database_path, 
            image_path = self.config.image_dir, 
            camera_model = self.config.camera_model,
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
            output_path = self.config.output_dir)

    def estimate_poses(self):
        self._extract_features()
        self._match_features()
        maps = self._incremental_mapping()
        # Write the output to the output directory, i.e. cameras, images and points3D
        maps[0].write(self.config.output_dir)

        # TODO: Check why it writes the output twice

        

        