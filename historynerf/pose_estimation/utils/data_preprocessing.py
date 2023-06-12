from pathlib import Path
import random
from typing import Optional

def undersample_fromlist(
        output_dir: Path, 
        image_list: list,
        rnd_seed: Optional[int]=None):
    '''
    Prepare the output directory for when an undersample list is provided. For sanity check, it saves the new image list in a txt file.
    '''
    output_dir = Path(output_dir) / f"chosen_nsamples{len(image_list)}"
    if output_dir.exists():
        output_dir = Path(output_dir) / f"chosen_nsamples{len(image_list)}_{rnd_seed}"
    output_dir.mkdir()
    with open(str(output_dir / f"undersample_list.txt"), "w") as f:
        f.write("\n".join(image_list))

    return image_list, output_dir

def undersample_images(
        image_path: Path, 
        output_dir: Path,
        sample_size: int,
        rnd_seed: Optional[int]=None):
    '''
    Unsersample the images in the image list. For sanity check, it saves the new image list in a txt file.
    '''
    image_list = [str(i.name) for i in Path(image_path).iterdir()]
    if rnd_seed:
        random.Random(rnd_seed).shuffle(image_list)
    undersample_list = image_list[:sample_size]
    output_dir = Path(output_dir) / f"seed{rnd_seed}_nsamples{sample_size}"
    output_dir.mkdir()

    with open(str(output_dir / f"undersample_list.txt"), "w") as f:
        f.write("\n".join(undersample_list))
    return undersample_list, output_dir

def undersample_video_frames(
        image_path: Path, 
        output_dir: Path,
        step_size: int,):
    '''
    Undersample the sequential frames of a video. Take a frame every step_size frames. For sanity check, it saves the new image list in a txt file.
    '''
    image_list = sorted([str(i.name) for i in Path(image_path).iterdir()])
    undersample_list = image_list[::step_size]

    output_dir = Path(output_dir) / f"step_size{step_size}"
    output_dir.mkdir()

    with open(str(output_dir / f"undersample_list.txt"), "w") as f:
        f.write("\n".join(undersample_list))
    return undersample_list, output_dir