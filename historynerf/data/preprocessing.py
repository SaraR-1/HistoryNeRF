import cv2
from pathlib import Path
import random
import shutil
from typing import Optional


from historynerf.config import PreprocessingConfig



# Create a class for preprocessing the data
class Preprocessing:
    def __init__(
            self,
            config: PreprocessingConfig,

    ):
        self.config = config
        self.initialize()
    
    def initialize(self):
        '''
        Creates output directories within the specified output directory.
        '''
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=self.config.overwrite_output)


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
        if self.config.sampling.sample_size:
            image_list = [str(i.name) for i in Path(self.config.input_dir).iterdir()]

            if self.config.sampling.rnd_seed:
                random.Random(self.config.sampling.rnd_seed).shuffle(image_list)
            undersample_list = image_list[:self.config.sampling.sample_size]

            return undersample_list
        return []
    
    def undersample_sequential(self, 
                               frames_folder=True):
        '''
        Undersample the sequential frames of a video. Take a frame every step_size frames. 
        '''
        if self.config.sampling.video_sample_step:
            input_dir = Path(self.config.input_dir) / "frames" if frames_folder else Path(self.config.input_dir)
            image_list = sorted([str(i.name) for i in Path(input_dir).iterdir()])
            undersample_list = image_list[::self.config.sampling.video_sample_step]
            return undersample_list
        return []

    def write_undersample_list(self, undersample_list):
        '''
        Write the undersample list to a file
        '''
        with open(Path(self.config.output_dir) / 'undersample_list.txt', 'w') as f:
            for item in undersample_list:
                f.write("%s\n" % item)

    def noise(self):
        pass

    def write_images(self):
        '''
        Copy the images in the undersample list to the output directory
        '''
        # Check if the input directory is a folder of images or a video 
        if self.config.input_dir.is_dir():
            undersample_list = self.undersample()
        else:
            self.video_to_frames()
            undersample_list = self.undersample_sequential(frames_folder=True)

        self.write_undersample_list(undersample_list)

        for image in undersample_list:
            image_path = Path(self.config.input_dir) / image
            # Copy the image to the output directory
            shutil.copy(image_path, Path(self.config.output_dir) / image)