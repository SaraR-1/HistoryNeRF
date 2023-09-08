from typing import List
from pathlib import Path
import random
import cv2
import shutil
from historynerf.config import DataPreparationConfig

class DataPreparation:
    def __init__(
        self, 
        config: DataPreparationConfig) -> None:
        """
        Initialize the DataPreparation class.

        Args:
            config (DataPreparationConfig): Configuration object.
        """
        self.config = config
        # Skip saving the images if no data preparation (sampling, resizing, frames from video) has been done
        self.skip_save = False
        self.initialize()

    def initialize(self) -> None:
        """Create output directories within the specified output directory."""
        output_dir = Path(self.config.output_dir) / "images"
        output_dir.mkdir(exist_ok=self.config.overwrite_output, parents=True)
        self.config.output_dir = output_dir

    def video_to_frames(self) -> int:
        """Extract frames from a video and save them in a folder."""
        frame_dir = Path(self.config.output_dir) / "frames"
        frame_dir.mkdir(exist_ok=self.config.overwrite_output)
        capture = cv2.VideoCapture(self.config.input_dir)
        frame_nr = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            cv2.imwrite(str(frame_dir / f'{frame_nr}.jpg'), frame)
            frame_nr += 1
        capture.release()
        print(f"Number of frames found: {frame_nr}") 

    def _sample_list(self, item_list: List[str], force_sequential: bool = False) -> List[str]:
        """Sample the items from a list based on the configuration."""
        if self.config.sampling.sample_size and not force_sequential:
            if self.config.sampling.rnd_seed:
                random.Random(self.config.sampling.rnd_seed).shuffle(item_list)
            return item_list[self.config.sampling.sequential_sample_start:self.config.sampling.sample_size+self.config.sampling.sequential_sample_start]
        elif self.config.sampling.sequential_sample_step:
            return item_list[::self.config.sampling.sequential_sample_step]
        return item_list
    
    def undersample(self) -> List[str]:
        """Undersample the images in the image list."""
        image_list = sorted([str(i.name) for i in Path(self.config.input_dir).iterdir()])
        return self._sample_list(image_list)

    def undersample_video(self, frames_folder: bool = True) -> List[Path]:
        """Undersample the sequential frames of a video."""
        frames_dir = Path(self.config.output_dir) / "frames" if frames_folder else Path(self.config.output_dir)
        # Overwrite the input direction to be pointing to the frames rather then the video
        self.config.input_dir = frames_dir
        file_names = [Path(i.name) for i in frames_dir.iterdir()]
        sorted_files = sorted(file_names, key=lambda x: int(x.stem))
        return self._sample_list(sorted_files, force_sequential=True)

    def write_undersample_list(self, undersample_list: List[str]) -> None:
        """Write the undersample list to a file."""
        with open(Path(self.config.output_dir) / 'undersample_list.txt', 'w') as f:
            for item in undersample_list:
                f.write("%s\n" % item)

    def save_images(self) -> None:
        """Copy or resize the images in the undersample list to the output directory."""
        video_flag = not Path(self.config.input_dir).is_dir()
        if video_flag:
            self.video_to_frames()
            undersample_list = self.undersample_video()
        else:
            undersample_list = self.undersample()
            self.skip_save = not bool(self.config.sampling.sample_size or self.config.sampling.sequential_sample_step)
        self.write_undersample_list(undersample_list)
        
        if not self.skip_save:
            for image in undersample_list:
                image_path = Path(self.config.input_dir) / image
                self._save_or_resize_image(image, image_path)
            # If input data is video, delete the temporary repository with all frames (unless specified otherwise)
            if video_flag:
                shutil.rmtree(Path(self.config.output_dir) / "frames")

    def _save_or_resize_image(self, image: str, image_path: Path) -> None:
        """Save or resize the image based on the configuration."""
        if self.config.resize:
            # Read and resize the image, then save it
            img = cv2.imread(str(image_path))
            img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]), cv2.INTER_AREA)
            cv2.imwrite(str(Path(self.config.output_dir) / image), img)
        else:
            # Copy the image to the output director
            shutil.copy(image_path, Path(self.config.output_dir) / image)
