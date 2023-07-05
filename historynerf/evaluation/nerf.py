from pathlib import Path
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json

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
    def __init__(self, config_path: str, camera_path: str, gt_images_dir: str=None, output_dir: str=None):
        self.data_manager = DataManager(config_path, camera_path, gt_images_dir)

        self.output_dir = output_dir
        self.images_path, self.pred_images, self.gt_images = self.get_data()
        
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

        return images_path, pred_images, gt_images

    def compute_metrics(self):
        '''
        Compute metrics for a given image and ground truth image
        '''
        psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3])
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        psnr_values = psnr(self.pred_images, self.gt_images)
        ssim_values = ssim(self.pred_images, self.gt_images)
        lpips_values = Tensor([lpips(self.pred_images[i].unsqueeze(0).float(), self.gt_images[i].unsqueeze(0).float()).item() for i in range(self.pred_images.shape[0])])

        # Print mean values in a nice format
        for metric, metric_name in zip([psnr_values, ssim_values, lpips_values], ["PSNR", "SSIM", "LPIPS"]):
            print(f"{metric_name}: {metric.mean().item():.4f}")

        # Save metrics vectors in a file 
        metrics_dict = {"PSNR": psnr_values.tolist(), "SSIM": ssim_values.tolist(), "LPIPS": lpips_values.tolist()}
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f)
    
    def save_rendered(self):
        '''
        Save all rendered (predict) images in a folder
        '''
        (self.output_dir / self.images_path[0].parent).mkdir(parents=True, exist_ok=True)
        for i in range(self.pred_images.shape[0]):
            plt.imsave(self.output_dir / self.images_path[i], self.pred_images[i].permute(1,2,0).numpy())
        

if __name__ == "__main__":
    # Arguments
    config_path = Path("/workspace/data/bridge_of_sighs/output/gold_standard/nerf/slick-swan/nerfacto/2023-07-04_141837/config.yml")
    camera_path = Path("/workspace/data/bridge_of_sighs/data/test/transforms.json")
    gt_images_dir = Path("/workspace/data/bridge_of_sighs/data/test")
    output_dir = Path("/workspace/data/bridge_of_sighs/prova")

    evaluator = Evaluator(config_path, camera_path, gt_images_dir, output_dir)
    evaluator.save_rendered()
    evaluator.compute_metrics()



# python3 historynerf/evaluation/nerf.py 