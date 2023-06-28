from pathlib import Path
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
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


class DataManager:
    def __init__(self, config_path: str, camera_path: str, gt_images_dir: str=None):
        self.config, self.pipeline, _, self.step = eval_setup(config_path=config_path, eval_num_rays_per_chunk=None, test_mode="test")
        if not gt_images_dir:
            gt_images_dir = self.config.data

        self.camera_path = camera_path
        self.gt_images_dir = gt_images_dir
        self.cameras_json = load_from_json(camera_path)
        self.camera_type = CAMERA_MODEL_TO_TYPE[self.cameras_json["camera_model"]]
        self.frames = self.cameras_json["frames"]    
        self.dataparser_config = self.pipeline.datamanager.dataparser.config

    def process_poses(self, frames):
        '''
        Process poses from COLMAP to NerfStudio format.
        '''
        images_path, poses = zip(*[[Path(f["file_path"]), f["transform_matrix"]] for f in frames])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        poses, _ = camera_utils.auto_orient_and_center_poses(
                poses,
                method=self.dataparser_config.orientation_method,
                center_method=self.dataparser_config.center_method,
            )
        
        if self.dataparser_config.auto_scale_poses:
            scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
            scale_factor *= scale_factor
            poses[:, :3, 3] *= scale_factor

        return list(images_path), poses

    def read_image(self, path, downscale_factor):
        image = plt.imread(path)
        # Downscale the image
        image = dycheck_dataparser.downscale(image, int(1/downscale_factor)).astype(np.float64)
        # Normalize the image
        image /=  255.0

        return image


class Evaluator:
    def __init__(self, config_path: str, camera_path: str, gt_images_dir: str=None):
        self.data_manager = DataManager(config_path, camera_path, gt_images_dir)
        
    def get_data(self):
        '''
        Compute metrics for a given image and ground truth image
        '''
        width, height, fx, fy, cx, cy = [self.data_manager.cameras_json[i] for i in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']]

        # List of parameters to fetch
        params = ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']
        # Fetch parameters using list comprehension and dictionary get method
        distortion_params = {param: self.data_manager.cameras_json.get(param, 0.0) for param in params}
        # Pass distortion_params as a dictionary to the function
        distortion_params = camera_utils.get_distortion_params(**distortion_params)

        downscale_factor = self.data_manager.dataparser_config.downscale_factor
        max_dim_flag = max(width, height) > 1600
        if downscale_factor is None:
            downscale_factor = 0.5 if max_dim_flag else 1.0

        images_path, processed_poses = self.data_manager.process_poses(self.data_manager.frames)
        
        cameras = Cameras(
            width=width, 
            height=height,
            fx=fx, fy=fy, 
            cx=cx, 
            cy=cy, 
            camera_to_worlds=processed_poses, 
            camera_type=self.data_manager.camera_type, 
            distortion_params=distortion_params)

        cameras.rescale_output_resolution(downscale_factor)

        pred_images, _ = render_trajectory(
            pipeline=self.data_manager.pipeline,
            cameras=cameras,
            rgb_output_name = "rgb",
            depth_output_name = "depth",
            rendered_resolution_scaling_factor = 1.0,
            )

        pred_images = torch.tensor(np.array(pred_images)).permute(0, 3, 1, 2)

        # Read and process ground truth images
        gt_images = [self.data_manager.read_image(self.data_manager.gt_images_dir / img_path, downscale_factor) for img_path in images_path]
        gt_images = torch.tensor(np.array(gt_images)).permute(0, 3, 1, 2)

        return pred_images, gt_images

    def compute_metrics(self):
        '''
        Compute metrics for a given image and ground truth image
        '''
        psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3])
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        pred_images, gt_images = self.get_data()

        psnr_values = psnr(pred_images, gt_images)
        ssim_values = ssim(pred_images, gt_images)
        lpips_values = Tensor([lpips(pred_images[i].unsqueeze(0).float(), gt_images[i].unsqueeze(0).float()).item() for i in range(pred_images.shape[0])])

        return psnr_values, ssim_values, lpips_values

if __name__ == "__main__":
    # Arguments
    config_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/nerf/processed_data/nerfacto/2023-06-22_145009/config.yml")
    camera_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data/transforms.json")
    gt_images_dir = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data")

    evaluator = Evaluator(config_path, camera_path, gt_images_dir)
    psnr_values, ssim_values, lpips_values = evaluator.compute_metrics()
    print(psnr_values.mean(), ssim_values.mean(), lpips_values.mean())
