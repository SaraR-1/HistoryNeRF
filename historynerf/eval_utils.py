import cv2
import numpy as np
from PIL import Image
from torch import Tensor
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.io import load_from_json
from nerfstudio.exporter.exporter_utils import render_trajectory

from pathlib import Path


def process_poses(frames, dataparser_config):
    '''
    Process poses from COLMAP to NerfStudio format.
    '''
    poses = [f["transform_matrix"] for f in frames]
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

    return poses
    

if __name__ == "__main__":
    # Arguments
    config_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/nerf/processed_data/nerfacto/2023-06-22_145009/config.yml")
    camera_path = Path("../data/bridge_of_sighs/output_temp/every5frames_III/processed_data/transforms.json")

    # Load config file
    config, pipeline, _, step = eval_setup(config_path=config_path, eval_num_rays_per_chunk=None, test_mode="test")
    # num_rays_per_chunk = config.viewer.num_rays_per_chunk
    # print(f"Number Ray Per Chunk: {num_rays_per_chunk}")
    # assert self.viewer.num_rays_per_chunk == -1
    # Or load_model(loaded_state: Dict[str, Any]) from nerfstudio.models.base_model.Model.load_model
    # breakpoint()
    # loaded_state = torch.load(checkpoint_path, map_location="cpu")
    # model = Model().load_model(loaded_state=loaded_state)

    # TODO: create bundle from cameras
    cameras_json = load_from_json(camera_path)
    camera_type = CAMERA_MODEL_TO_TYPE[cameras_json["camera_model"]]
    # This is always fixed when transforming from COLMAP to NerfStudio - is it the "distortion_params" argument in Cameras?
    applied_transform = cameras_json["applied_transform"]
    # k1, k2, p1, p2 (other eventual ks) are the radial (k) and tangential (p) distortion coefficients
    # "distortion_params" argument in Cameras takes the k1, ... , k6 OpenCV radial distortion coefficients

    frames = cameras_json["frames"]    
    dataparser_config = pipeline.datamanager.dataparser.config

    # Naive example
    # c2w = [Tensor(frames[0]['transform_matrix']), Tensor(frames[1]['transform_matrix'])]
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
    if downscale_factor is None:
        downscale_factor = 0.5 if max(width, height) > 1600 else 1.0

    processed_poses = process_poses(frames, dataparser_config)

    for pose_idx in range(0, processed_poses.shape[0]):
        # c2w = Tensor(frames[frame_num]['transform_matrix'])[None, :3, :]
        # camera = pipeline.datamanager.train_dataset.cameras[0]

        # c2w = Tensor(applied_transform) @ Tensor(frames[frame_num]['transform_matrix'])

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
        
        image = image[0]
        from matplotlib import pyplot as plt
        plt.imsave(f"./test_{pose_idx}.png", image)



