import os
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
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers import dycheck_dataparser
from nerfstudio.models.base_model import Model
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.io import load_from_json
from nerfstudio.exporter.exporter_utils import render_trajectory

def extract_cameras(camera_path):
    cameras_json = load_from_json(camera_path)
    camera_type = CAMERA_MODEL_TO_TYPE[cameras_json["camera_model"]]
    frames = cameras_json["frames"]

    return cameras_json, camera_type, frames



class DataManager:
    def __init__(self, config_path: Path, camera_path_test: Path, gt_images_dir: str=None):
        self.config, self.pipeline, _, self.step = eval_setup(config_path=config_path, eval_num_rays_per_chunk=None, test_mode="test")
        
        if not gt_images_dir:
            gt_images_dir = self.config.data

        self.gt_images_dir = gt_images_dir

        self.cameras_json_test, self.camera_type_test, self.frames_test = extract_cameras(camera_path_test)
        # self.cameras_json_train, self.camera_type_train, self.frames_train = extract_cameras(camera_path_train)

        self.dataparser_config = self.pipeline.datamanager.dataparser.config

        dataparser_transforms = load_from_json(Path(config_path).parent / "dataparser_transforms.json")
        self.transform = torch.tensor(dataparser_transforms["transform"])
        self.scale = dataparser_transforms["scale"]


    def read_image(self, path, downscale_factor):
        image = plt.imread(path)
        # Downscale the image
        image = dycheck_dataparser.downscale(image, int(1/downscale_factor)).astype(np.float64)
        # Normalize the image
        image /=  255.0

        return image

    def process_poses_testscale(self, poses):
        poses, _ = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.dataparser_config.orientation_method,
            center_method=self.dataparser_config.center_method,
        )
        
        if self.dataparser_config.auto_scale_poses:
            scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
            scale_factor *= scale_factor
            poses[:, :3, 3] *= scale_factor

        return poses

    def process_poses_trainscale(self, poses):
        self.transform = torch.index_select(self.transform, 1, torch.LongTensor([1,0,2,3]))
        self.transform[:,2] *= -1

        poses = self.transform @ poses
        poses[:, :3, 3] *= self.scale**2

        return poses

    def process_poses(self):
        '''
        Process poses from COLMAP to NerfStudio format.
        '''
        images_path, poses = zip(*[[Path(f["file_path"]), f["transform_matrix"]] for f in self.frames_test])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        poses_testscale = self.process_poses_testscale(poses)
        poses_trainscale = self.process_poses_trainscale(poses)

        # _ , poses_train = zip(*[[Path(f["file_path"]), f["transform_matrix"]] for f in self.frames_train])
        # poses_train = torch.from_numpy(np.array(poses_train).astype(np.float32))

        # poses_train, transform_train = camera_utils.auto_orient_and_center_poses(poses_train)
        # scale_factor_train = 1.0 / float(torch.max(torch.abs(poses_train[:, :3, 3])))

        # poses = transform_train @ poses
        # poses[:, :3, 3] *= scale_factor_train


        return list(images_path), poses_testscale, poses_trainscale

class NerfEvaluator:
    def __init__(self, config_path: Path, camera_path_test: Path, gt_images_dir: Path=None, output_dir: Path=None):
        self.data_manager = DataManager(config_path, camera_path_test, gt_images_dir)

        self.config_path = config_path
        self.output_dir = output_dir
        self.images_path, self.pred_images_testscale, self.pred_images_trainscale, self.gt_images = self.get_data()

    def render_cameras(self, processed_poses):
        cameras = Cameras(
            width=self.camera_properties["w"], 
            height=self.camera_properties["h"],
            fx=self.camera_properties["fl_x"], fy=self.camera_properties["fl_y"], 
            cx=self.camera_properties["cx"], 
            cy=self.camera_properties["cy"], 
            camera_to_worlds=processed_poses, 
            camera_type=self.data_manager.camera_type_test, 
            distortion_params=self.distortion_params)

        cameras.rescale_output_resolution(self.downscale_factor)

        pred_images, _ = render_trajectory(
            pipeline=self.data_manager.pipeline,
            cameras=cameras,
            rgb_output_name = "rgb",
            depth_output_name = "depth",
            rendered_resolution_scaling_factor = 1.0,
            )

        pred_images = torch.tensor(np.array(pred_images)).permute(0, 3, 1, 2)
        return pred_images

        
    def get_data(self):
        '''
        Compute metrics for a given image and ground truth image
        '''
        # width, height, fx, fy, cx, cy = [self.data_manager.cameras_json_test[i] for i in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']]

        self.camera_properties = {i: self.data_manager.cameras_json_test[i] for i in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']}

        # List of parameters to fetch
        params = ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']
        # Fetch parameters using list comprehension and dictionary get method
        distortion_params = {param: self.data_manager.cameras_json_test.get(param, 0.0) for param in params}
        # Pass distortion_params as a dictionary to the function
        self.distortion_params = camera_utils.get_distortion_params(**distortion_params)

        self.downscale_factor = self.data_manager.dataparser_config.downscale_factor
        max_dim_flag = max(self.camera_properties["w"], self.camera_properties["h"]) > 1600
        if self.downscale_factor is None:
            self.downscale_factor = 0.5 if max_dim_flag else 1.0

        images_path, processed_poses_testscale, processed_poses_trainscale = self.data_manager.process_poses()
        
        pred_images_testscale = self.render_cameras(processed_poses_testscale)
        pred_images_trainscale = self.render_cameras(processed_poses_trainscale)

        # Read and process ground truth images
        gt_images = [self.data_manager.read_image(self.data_manager.gt_images_dir / img_path, self.downscale_factor) for img_path in images_path]
        gt_images = torch.tensor(np.array(gt_images)).permute(0, 3, 1, 2)

        return images_path, pred_images_testscale, pred_images_trainscale, gt_images

    def compute_metrics_singlescale(self, pred_scale):
        '''
        Compute metrics for a given image and ground truth image
        '''
        psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3])
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if pred_scale == 'test':
            pred_images = self.pred_images_testscale
        elif pred_scale == 'train':
            pred_images = self.pred_images_trainscale

        psnr_values = psnr(pred_images, self.gt_images)
        ssim_values = ssim(pred_images, self.gt_images)
        lpips_values = Tensor([lpips(pred_images[i].unsqueeze(0).float(), self.gt_images[i].unsqueeze(0).float()).item() for i in range(pred_images.shape[0])])

        # Print mean values in a nice format
        for metric, metric_name in zip([psnr_values, ssim_values, lpips_values], ["PSNR", "SSIM", "LPIPS"]):
            print(f"{metric_name}: {metric.mean().item():.4f}")

        # Save metrics vectors in a file 
        metrics_dict = {f"PSNR {pred_scale}_scale": psnr_values.tolist(), f"SSIM {pred_scale}_scale": ssim_values.tolist(), f"LPIPS {pred_scale}_scale": lpips_values.tolist()}
        # Log metrics to wandb
        wandb.log(metrics_dict)

    def compute_metrics(self):
        self.compute_metrics_singlescale(pred_scale="test")
        self.compute_metrics_singlescale(pred_scale="train")
        
    def save_rendered_images_singlescale(self, pred_scale):
        '''
        Save all rendered (predict) images in a folder
        '''
        if pred_scale == 'test':
            pred_images = self.pred_images_testscale
        elif pred_scale == 'train':
            pred_images = self.pred_images_trainscale

        (self.output_dir / f"{pred_scale}_scale" / self.images_path[0].parent).mkdir(parents=True, exist_ok=True)
        for i in range(pred_images.shape[0]):
            plt.imsave(self.output_dir / f"{pred_scale}_scale" / self.images_path[i], pred_images[i].permute(1,2,0).numpy())
    
    def save_rendered_images(self):
        self.save_rendered_images_singlescale(pred_scale="test")
        self.save_rendered_images_singlescale(pred_scale="train")

    def save_rendered_video(self):
        '''
        Save rendered video - based on the training set
        '''
        output_dir_video = self.output_dir / "output.mp4"
        command = f"ns-render interpolate --load-config {self.config_path} --output-path {output_dir_video} --pose-source train"

        os.system(command)

    def save_rendered(self):
        self.save_rendered_images()
        self.save_rendered_video()
        