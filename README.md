## 1. Installation: Setup the environment
Use the provided docker image to install the dependencies for nerfstudio. Docker (get docker) and nvidia GPU drivers (get nvidia drivers), capable of working with CUDA 11.8, must be installed.

Then update the required dependencies:
```
pip install -e ./
```

## 2. NeRFStudio Preprocessing and Structure from motion
Usage: `ns-process-data images` `--data PATH --output-dir PATH [--verbose | --no-verbose] [--camera-type {perspective,fisheye,equirectangular}]
                              [--matching-method {exhaustive,sequential,vocab_tree}] [--sfm-tool {any,colmap,hloc}] [--refine-pixsfm | --no-refine-pixsfm]
                              [--feature-type {any,sift,superpoint,superpoint_aachen,superpoint_max,superpoint_inloc,r2d2,d2net-ss,sosnet,disk}]
                              [--matcher-type {any,NN,superglue,superglue-fast,NN-superpoint,NN-ratio,NN-mutual,adalam}] [--num-downscales INT]
                              [--skip-colmap | --no-skip-colmap] [--skip-image-processing | --no-skip-image-processing] [--colmap-model-path PATH] [--colmap-cmd STR]
                              [--images-per-equirect {8,14}] [--crop-factor FLOAT FLOAT FLOAT FLOAT] [--crop-bottom FLOAT] [--gpu | --no-gpu]
                              [--use-sfm-depth | --no-use-sfm-depth] [--include-depth-debug | --no-include-depth-debug]`

Explaining `any` defaults, specifically how sfm replaces the default parameters `any` by usable value:

```
if sfm_tool == "any":
    if (feature_type in ("any", "sift")) and (matcher_type in ("any", "NN")):
        sfm_tool = "colmap"
    else:
        sfm_tool = "hloc"
if sfm_tool == "colmap":
    if (feature_type not in ("any", "sift")) or (matcher_type not in ("any", "NN")):
        return (None, None, None)
    return ("colmap", "sift", "NN")
if sfm_tool == "hloc":
    if feature_type in ("any", "superpoint"):
        feature_type = "superpoint_aachen"

    if matcher_type == "any":
        matcher_type = "superglue"
    elif matcher_type == "NN":
        matcher_type = "NN-mutual"
```

## 3. Train a NeRF model Example
1. Train Gold Standard
```
python3 historynerf/run.py wandb_project=bridge_of_sighs data_preparation.input_dir=/workspace/data/bridge_of_sighs/output/gold_standard/processed_data data_preparation.output_dir=/workspace/data/bridge_of_sighs/output/gold_standard data_preparation.overwrite_output=True pose_estimation.skip_image_processing_flag=True pose_estimation.skip_colmap_flag=True pose_estimation.colmap_model_path=/workspace/data/bridge_of_sighs/output/gold_standard/processed_data/colmap/sparse/0 pose_estimation.matching_method=sequential nerf.train_split_fraction=1. nerf.pipeline.model.use_gradient_scaling=False
```

```
python3 historynerf/run.py wandb_project=bridge_of_sighs data_preparation.input_dir=/workspace/data/bridge_of_sighs/data/train/images data_preparation.output_dir=/workspace/data/bridge_of_sighs/output/alltrain data_preparation.overwrite_output=False
pose_estimation.matching_method=sequential nerf.train_split_fraction=1. nerf.pipeline.model.use_gradient_scaling=False

python3 historynerf/run.py wandb_project=bridge_of_sighs data_preparation.input_dir=/workspace/data/bridge_of_sighs/data/train/images data_preparation.output_dir=/workspace/data/bridge_of_sighs/output/every10frames data_preparation.overwrite_output=False data_preparation.sampling.video_sample_step=10 pose_estimation.matching_method=sequential nerf.train_split_fraction=1. nerf.pipeline.model.use_gradient_scaling=False 

python3 historynerf/run.py wandb_project=bridge_of_sighs data_preparation.input_dir=/workspace/data/bridge_of_sighs/data/train/images data_preparation.output_dir=/workspace/data/bridge_of_sighs/output/every10frames data_preparation.overwrite_output=False data_preparation.sampling.sequential_sample_step=10 pose_estimation.matching_method=sequential nerf.vis=wandb

python3 historynerf/run.py wandb_project=bridge_of_sighs data_preparation.input_dir=/workspace/data/bridge_of_sighs/output/every10frames/images data_preparation.output_dir=/workspace/data/bridge_of_sighs/output/every10frames data_preparation.overwrite_output=True pose_estimation.skip_colmap_flag=True pose_estimation.colmap_model_path=/workspace/data/bridge_of_sighs/output/every10frames/processed_data/colmap/sparse/0 nerf.vis=wandb
```

2. Split Into Train and Test
```
python3 historynerf/split_data.py --SplitDataConfig.camera_path /workspace/data/bridge_of_sighs/output/gold_standard/processed_data/transforms.json --SplitDataConfig.n 80 --SplitDataConfig.images_dir /workspace/data/bridge_of_sighs/output/gold_standard/processed_data/images --SplitDataConfig.output_dir /workspace/data/bridge_of_sighs/data
```


## 4. Render Video or Visualize Existing Run
First we must create a path for the camera to follow. This can be done in the viewer under the “RENDER” tab. Orient your 3D view to the location where you wish the video to start, then press “ADD CAMERA”. This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press “RENDER” which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and run the command to generate the video.

Given a pretrained model checkpoint, you can start the viewer by running:
`ns-viewer --load-config {outputs/.../config.yml}`