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
        pycolmap.extract_features(
            database_path = self.config["database_path"], 
            image_path = self.config["image_dir"], 
            camera_model = self.config["camera_model"],
        )
    
    def _match_features(self):
        if self.config["matching_method"] == "exhaustive":
            pycolmap.match_exhaustive(
                database_path = self.config["database_path"],
            )
        elif self.config["matching_method"] == "sequential":
            pycolmap.match_sequential(
                database_path = self.config["database_path"],
            )
        else:
            raise ValueError(f"Matching method {self.config['matching_method']} not supported")

    def estimate_poses(self):
        self._extract_features()
        self._match_features()

        

        