import cv2
from pathlib import Path
import random
import shutil
from typing import Optional

from historynerf.config import DataPreparationConfig

# Create a class for preprocessing the data
class DataPreparation:
    def __init__(
            self,
            config: DataPreparationConfig,

    ):
        self.config = config
        # Skip saving the images if no data preparation (sampling, resizing, frames from video) has been done
        self.skip_save = False       
        self.initialize()
    
    def initialize(self):
        '''
        Creates output directories within the specified output directory.
        '''
        output_dir = Path(self.config.output_dir) / "images"
        output_dir.mkdir(exist_ok=self.config.overwrite_output, parents=True)

        self.config.output_dir = output_dir


    def video_to_frames(self):
        '''
        Extract the frames from a video and save them in a folder.

            Example:
            dataset_path = Path("/bridge_of_sighs")
            video_path = dataset_path / "bridge_of_sighs.mp4"
            save_path = dataset_path / "frames"

            video_to_frames(video_path, save_path)
        '''
        frame_dir = Path(self.config.output_dir) / "frames"
        frame_dir.mkdir(exist_ok=self.config.overwrite_output)

        capture = cv2.VideoCapture(self.config.input_dir)
        frameNr = 0
        
        while (True):
            success, frame = capture.read()
            if success:
                cv2.imwrite(str(frame_dir / f'{frameNr}.jpg'), frame)
            else:
                break
            frameNr += 1
        capture.release()

        print(f"Number of frames found: {frameNr}")

    def undersample(self):
        '''
        Unsersample the images in the image list. If rnd_seed=None, then the sampling is non-random but based on closeness - requires some gold standard data for camera poses
        '''
        image_list = sorted([str(i.name) for i in Path(self.config.input_dir).iterdir()])
        if self.config.sampling.sample_size:
            if self.config.sampling.rnd_seed:
                random.Random(self.config.sampling.rnd_seed).shuffle(image_list)
            undersample_list = image_list[:self.config.sampling.sample_size]
            return undersample_list
        elif self.config.sampling.sequential_sample_step:
            undersample_list = image_list[::self.config.sampling.sequential_sample_step]
            return undersample_list
        return image_list

    def undersample_video(self, frames_folder=True):
        '''
        Undersample the sequential frames of a video. Take a frame every step_size frames. 
        '''
        frames_dir = Path(self.config.output_dir) / "frames" if frames_folder else Path(self.config.output_dir)
        # Overwrite the input direction to be pointing to the frames rather then the video
        self.config.input_dir = frames_dir    
        # Sort filenames numerically
        file_names = [Path(i.name) for i in Path(self.config.input_dir).iterdir()]
        sorted_files = sorted(file_names, key=lambda x: int(x.stem))

        if self.config.sampling.sequential_sample_step:
            undersample_list = sorted_files[::self.config.sampling.sequential_sample_step]
            return undersample_list
        return sorted_files

    def write_undersample_list(self, undersample_list):
        '''
        Write the undersample list to a file
        '''
        with open(Path(self.config.output_dir) / 'undersample_list.txt', 'w') as f:
            for item in undersample_list:
                f.write("%s\n" % item)

    def noise(self):
        pass

    def save_images(self):
        '''
        Copy the images in the undersample list to the output directory
        '''
        # Check if the input directory is a folder of images or a video
        video_flag = False    
        if Path(self.config.input_dir).is_dir():
            undersample_list = self.undersample()
            self.skip_save = self.config.sampling.sample_size is None and self.config.sampling.sequential_sample_step is None
        else:
            frames_folder = True
            video_flag = True  
            self.video_to_frames()
            undersample_list = self.undersample_video(frames_folder=frames_folder)
        self.write_undersample_list(undersample_list)

        if not self.skip_save:
            for image in undersample_list:
                image_path = Path(self.config.input_dir) / image
                if self.config.resize:
                    # Read and resize the image, then save it
                    img = cv2.imread(str(image_path))
                    img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]), cv2.INTER_AREA)
                    cv2.imwrite(str(Path(self.config.output_dir) / image), img)
                else:
                    # Copy the image to the output director
                    shutil.copy(image_path, Path(self.config.output_dir) / image)

        
            # If input data is video, delete the temporary repository with all frames (unless specified otherwise)
            if video_flag:
                frames_dir = Path(self.config.output_dir) / "frames" if frames_folder else Path(self.config.output_dir)
                shutil.rmtree(frames_dir)