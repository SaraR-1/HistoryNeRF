import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
import numpy as np
from historynerf.config import SplitDataConfig
from historynerf.utils import register_configs
from hydra.core.config_store import ConfigStore

from nerfstudio.utils.io import load_from_json


class SplitData:
    """
    Class to split the data into training and test sets.
    """
    def __init__(
        self, 
        config: SplitDataConfig
        ) -> None:
        self.cameras_json = load_from_json(config.camera_path)
        self.frames = self.cameras_json["frames"]
        self.n = config.n
        self.images_dir = config.images_dir
        self.output_dir = config.output_dir

        self.test_set = self.get_test_set()
        self.train_set = self.get_train_set()

    def get_test_set(self) -> List[str]:
        """
        Define a list of test images. Starting with a random image, take the next N farther away images.
        Returns:
            A list of test image filenames.
        """
        # Get the translation vectors
        transl_vect = np.array([np.array(f["transform_matrix"])[:3, 3] for f in self.frames])
        # Compute the pairwise distance between all the cameras
        dist = np.linalg.norm(transl_vect[:, None] - transl_vect, axis=-1)
        # Set the diagonal to inf
        np.fill_diagonal(dist, -np.inf)
        # Take the first image randomly as the starting point
        start_idx = np.random.randint(0, len(self.frames))
        selected_idx = [start_idx]
        # Take the N further images
        for _ in range(self.n - 1):
            selected_idx.append(np.argmax(np.mean(dist[selected_idx], axis=0)))
        return [Path(self.frames[i]['file_path']).name for i in selected_idx]

    def get_train_set(self) -> List[str]:
        """
        The training set is the complement of the test set.
        Returns:
            A list of train image filenames.
        """
        return [Path(f['file_path']).name for f in self.frames if Path(f['file_path']).name not in self.test_set]

    def create_data_folder(self, split: str = "test") -> None:
        """
        Create a new folder and copy images to a new location.
        """
        data_split = self.test_set if split == "test" else self.train_set
        (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        for f in data_split:
            shutil.copy(self.images_dir / f, self.output_dir / split / "images" / f)

    def create_camera_file(self, split: str = "test") -> None:
        """
        Create a new camera file. All the keys are the same, but the frame values are different,
        based on the test and train sets.
        """
        data_split = self.test_set if split == "test" else self.train_set
        camera_filename = self.output_dir / split / "transforms.json"
        camera_dict = {k: v for k, v in self.cameras_json.items() if k != "frames"}
        new_frames = [frame for frame in self.frames for f in data_split if Path(frame["file_path"]).name == f]
        camera_dict["frames"] = new_frames

        with open(camera_filename, 'w') as json_file:
            json.dump(camera_dict, json_file, indent=4)

    def create_data(self) -> None:
        """
        Create the data for the test and train sets.
        """
        for split in ["test", "train"]:
            self.create_data_folder(split)
            self.create_camera_file(split)

root_dir = Path(__file__).parents[1]

CONFIGURATIONS = [
    ('base', 'base_splitdata', SplitDataConfig),
]

register_configs(CONFIGURATIONS)
@hydra.main(config_path=str(root_dir / "configs"), config_name="parent_splitdata", version_base="1.1")
def main(cfg: SplitDataConfig) -> None:
    data_split = SplitData(cfg)
    data_split.create_data()

if __name__ == "__main__":
    main()
