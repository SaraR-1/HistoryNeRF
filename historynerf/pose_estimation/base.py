from historynerf.pose_estimation.configs import PoseEstimationConfig
from abc import ABC, abstractmethod
from pathlib import Path

class PoseEstimator(ABC):
    def __init__(
        self, 
        config: PoseEstimationConfig
    ):
        self.config = config
        self.initialize()

    def initialize(self):
        #  Creates output directories (required by COLMAP) within the specified output directory
        self.output = Path(self.config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "mvs").mkdir(parents=True, exist_ok=True)
        self.db_path = (self.output / self.config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


    @abstractmethod
    def estimate_poses(self):
        pass
    


