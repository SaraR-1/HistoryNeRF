import numpy as np
from pathlib import Path
import shutil
import json

from nerfstudio.utils.io import load_from_json


class SplitData:
    '''
    Split the data into training and test sets.
    '''
    def __init__(self, camera_path: Path, n: int, images_dir: Path, output_dir: Path):
        self.cameras_json = load_from_json(camera_path)
        self.frames = self.cameras_json["frames"]
        self.n = n
        self.images_dir = images_dir
        self.output_dir = output_dir

        self.test_set = self.get_test_set()
        self.train_set = self.get_train_set()
    
    def get_test_set(self):
        '''
        Define a list of test images. Starting with a random image, take the N further images.
        '''
        # Get the translation vectors
        transl_vect = np.array([np.array(f["transform_matrix"])[:3, 3] for f in self.frames])
        # Compute the pairwise distance between all the cameras
        dist = np.linalg.norm(transl_vect[:, None] - transl_vect, axis=-1)
        # Set the diagonal to inf
        np.fill_diagonal(dist, -np.inf)
        # Take the first image randomly as the starting point
        start_idx = np.random.randint(0, len(self.frames))
        # Take the N further images
        selected_idx = [start_idx]
        for i in range(self.n - 1):
            selected_idx.append(np.argmax(np.mean(dist[selected_idx], axis=0)))

        test_frames = [Path(self.frames[i]['file_path']).name for i in selected_idx]

        return test_frames
    
    def get_train_set(self):
        '''
        Training set is the complement of the test set.
        '''
        return [Path(f['file_path']).name for f in self.frames if Path(f['file_path']).name not in self.test_set]

    def create_data_folder(self, split="test"):
        '''
        Create a new folder and copy images to a new location
        '''
        data_split = self.test_set if split == "test" else self.train_set

        # Create the subfolders
        (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        # Copy the images
        for f in data_split:
            shutil.copy(self.images_dir / f, self.output_dir / split)
        
    def create_camera_file(self, split="test"):
        '''
        Create a new camera file. All the keys are the same, but the values of the frames are different - based on the test and train sets.
        '''
        data_split = self.test_set if split == "test" else self.train_set
        camera_filename = self.output_dir / split / "transforms.json"

        camera_dict = {k: v for k, v in self.cameras_json.items() if k != "frames"}
        # Copy the frames
        new_frames = [frame for frame in self.frames for f in data_split if Path(frame["file_path"]).name == f]
        # for f in data_split:
        #     new_frames.append([frame for frame in self.frames if frame["file_path"] == f])
        camera_dict["frames"] = new_frames

        with open(camera_filename, 'w') as json_file:
            json.dump(camera_dict, json_file, indent=4)

    def create_data(self):
        '''
        Create the data for the test and train sets.
        '''
        for split in ["test", "train"]:
            self.create_data_folder(split)
            self.create_camera_file(split)

if __name__ == "__main__":
    camera_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data/transforms.json")
    images_dir = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data/images")
    output_dir = Path("../data/bridge_of_sighs/prova_split")
    data_split = SplitData(camera_path, 10, images_dir, output_dir)
    data_split.create_data()



