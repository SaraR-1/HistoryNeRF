import cv2
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torch import Tensor
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers import dycheck_dataparser
from nerfstudio.models.base_model import Model
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.io import load_from_json
from nerfstudio.exporter.exporter_utils import render_trajectory


def process_poses(frames, dataparser_config):
    '''
    Process poses from COLMAP to NerfStudio format.
    '''
    images_path, poses = zip(*[[Path(f["file_path"]), f["transform_matrix"]] for f in frames])
    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    orientation_method = dataparser_config.orientation_method
    center_method = dataparser_config.center_method
    auto_scale_poses = dataparser_config.auto_scale_poses
    scale_factor = dataparser_config.scale_factor


    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=center_method,
        )
    
    scale_factor = 1.0
    if auto_scale_poses:
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    scale_factor *= scale_factor

    poses[:, :3, 3] *= scale_factor

    return list(images_path), poses

def switch_image(img):
    '''
    Switch image from [H, W, C] to [1, C, H, W] for metrics computations
    '''
    return torch.moveaxis(torch.from_numpy(img), -1, 0)[None, ...]

if __name__ == "__main__":
    # Arguments
    config_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/nerf/processed_data/nerfacto/2023-06-22_145009/config.yml")
    camera_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data/transforms.json")

    # Load config file
    config, pipeline, _, step = eval_setup(config_path=config_path, eval_num_rays_per_chunk=None, test_mode="test")
    gt_images_path = config.data

    cameras_json = load_from_json(camera_path)
    camera_type = CAMERA_MODEL_TO_TYPE[cameras_json["camera_model"]]

    # This is always fixed when transforming from COLMAP to NerfStudio - is it the "distortion_params" argument in Cameras?
    applied_transform = cameras_json["applied_transform"]
    # k1, k2, p1, p2 (other eventual ks) are the radial (k) and tangential (p) distortion coefficients
    # "distortion_params" argument in Cameras takes the k1, ... , k6 OpenCV radial distortion coefficients

    frames = cameras_json["frames"]    
    dataparser_config = pipeline.datamanager.dataparser.config

    width, height, fx, fy, cx, cy = [cameras_json[i] for i in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']]

    distortion_params = camera_utils.get_distortion_params(
        k1 = cameras_json["k1"] if "k1" in cameras_json else 0.0,
        k2 = cameras_json["k2"] if "k2" in cameras_json else 0.0,
        k3 = cameras_json["k3"] if "k3" in cameras_json else 0.0,
        k4 = cameras_json["k4"] if "k4" in cameras_json else 0.0,
        p1 = cameras_json["p1"] if "p1" in cameras_json else 0.0,
        p2 = cameras_json["p2"] if "p2" in cameras_json else 0.0,
    )

    downscale_factor = dataparser_config.downscale_factor
    max_dim_flag = max(width, height) > 1600
    if downscale_factor is None:
        downscale_factor = 0.5 if max_dim_flag else 1.0

    height_downscale, width_downscale = int(height * downscale_factor), int(width * downscale_factor)

    images_path, processed_poses = process_poses(frames, dataparser_config)


    images_pred = torch.empty(0, 3, height_downscale, width_downscale)
    images_gt = torch.empty(0, 3, height_downscale, width_downscale)

    for pose_idx in range(0, processed_poses.shape[0]):

        camera = Cameras(
            width=width, 
            height=height, 
            fx=fx, 
            fy=fy, 
            cx=cx, 
            cy=cy, 
            camera_to_worlds=processed_poses[pose_idx], 
            camera_type=camera_type, 
            distortion_params=distortion_params)
        
        camera.rescale_output_resolution(downscale_factor)

        image, _ = render_trajectory(
            pipeline=pipeline,
            cameras=camera,
            rgb_output_name = "rgb",
            depth_output_name = "depth",
            rendered_resolution_scaling_factor = 1.0,
        )
        
        # image = image[0]
        # images_pred = torch.cat((images_pred, switch_image(image[0].astype(np.float64))), dim=0)
        images_pred = torch.cat((images_pred, switch_image(image[0])), dim=0)

        # plt.imsave(f"./test_{pose_idx}.png", image)

        # filename_gt = Path("images_2") / Path(images_path[pose_idx]).name if max_dim_flag else images_path[pose_idx]
        image_gt = plt.imread(str(gt_images_path / images_path[pose_idx]), )
        # Downscale the image
        image_gt = dycheck_dataparser.downscale(image_gt, int(1/downscale_factor)).astype(np.float64)
        # Normalize the image
        image_gt /=  255.0
        images_gt = torch.cat((images_gt, switch_image(image_gt)), dim=0)
        
    psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3])
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
    # lpips does not allow reduction='none'
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
    
    psnr_values = psnr(images_pred, images_gt)
    ssim_values = ssim(images_pred, images_gt)
    lpips_values = Tensor([lpips(images_pred[i].unsqueeze(0).float(), images_gt[i].unsqueeze(0).float()).item() for i in range(images_pred.shape[0])])
    print(psnr_values)
    print(ssim_values)
    print(lpips_values)

    breakpoint()
    # Check on the validation set:
    validation_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
    validation_filenames = [Path(i).name for i in validation_filenames]
    # Mask images_path contained in validation_filenames
    validation_mask = [i.name in validation_filenames for i in images_path]

    print(psnr_values[validation_mask].mean())
    print(ssim_values[validation_mask].mean())
    print(lpips_values[validation_mask].mean())


    # breakpoint()
