from typing import Union, List, Tuple
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import wandb
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit as st
import json
import subprocess
from typing import List
from historynerf.config import NeRFConfig, NeRFPipelineConfig, MachineConfig, NeRFDataManagerConfig, NeRFModelConfig, NeRFCameraOptimizerConfig, EvaluationConfig, AlignmentEvaluationConfig, PoseEstimationConfig

import torch
from torchvision.transforms.functional import to_tensor
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load_wandb_data(entity: str, project: str, state: List[str] = ["finished"]) -> pd.DataFrame:
    '''
    Load the data from W&B and return a dataframe.
    '''
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}") 

    data = []
    for run in runs:
        if run.state in state:
            try:
                if isinstance(run.config["nerf"], str):
                    config_nerf = eval(run.config["nerf"])
                    config_pose = eval(run.config["pose_estimation"])
                    use_gradient_scaling = config_nerf.pipeline.model.use_gradient_scaling
                    max_num_iterations =config_nerf.max_num_iterations
                    method_name = config_nerf.method_name
                    colmap_model_path = config_pose.colmap_model_path if config_pose.colmap_model_path is not None else "",
                    gt_images_dir = eval(run.config["evaluation"]).gt_images_dir
                else:
                    use_gradient_scaling = run.config["nerf"]["pipeline"]["model"]["use_gradient_scaling"]
                    max_num_iterations = run.config["nerf"]["max_num_iterations"]
                    method_name = run.config["nerf"]["method_name"]
                    colmap_model_path = run.config["pose_estimation"]["colmap_model_path"] if run.config["pose_estimation"]["colmap_model_path"] is not None else "",
                    gt_images_dir = run.config["evaluation"]["gt_images_dir"]
                    
                data.append({
                    "name": run.name,
                    "use_gradient_scaling": use_gradient_scaling,
                    "max_num_iterations": max_num_iterations,
                    "method_name": method_name,
                    "colmap_model_path": colmap_model_path,
                    "gt_images_dir": gt_images_dir,
                    "output_path_nerf": run.config["output_path_nerf"],
                    "Training Sample Size": run.summary["Training Sample Size"], 
                    "output_config_path": run.config["output_config_path"],
                    "SSIM test_scale": run.summary["SSIM test_scale"],
                    "LPIPS test_scale": run.summary["LPIPS test_scale"],
                    "PSNR test_scale": run.summary["PSNR test_scale"],
                    "SSIM train_scale": run.summary["SSIM train_scale"],
                    "LPIPS train_scale": run.summary["LPIPS train_scale"],
                    "PSNR train_scale": run.summary["PSNR train_scale"],})
            except:
                st.write(run.name)
            
    df = pd.DataFrame(data)
    return df 

def filter_dataset_colmap(df: pd.DataFrame, colmap: str) -> pd.DataFrame:
    """
    Filter the dataframe to select only the experiments with the selected colmap model
    """
    fixed_colmap_mask = df["colmap_model_path"].str.contains("processed_data_fixedcolmap/colmap/sparse/0")

    if colmap == "estimated":
        df = df[~fixed_colmap_mask]
    elif colmap == "fixed":
        df = df[fixed_colmap_mask]
    else:
        raise ValueError(f"Unsupported colmap model: {colmap}")
    return df

def load_media(file_path: Union[str, Path], media_type: str) -> Union[bytes, Image.Image]:
    """
    Load a media file (video or image) and return the bytes or Image object
    """
    with open(str(file_path), 'rb') as file:
        if media_type == 'video':
            return file.read()
        elif media_type == 'image':
            return Image.open(file_path)
        else:
            raise ValueError(f'Unsupported media type: {media_type}')

def create_plot(df: pd.DataFrame, metric: str, metric_title:str, colmap_filter: str, x_name: str = "Training Sample Size", filter: bool=True):
    if filter:
        df = filter_dataset_colmap(df, colmap_filter)
        
    df_p = df.explode(metric)
    x = df_p[x_name].astype(str)
    y = df_p[metric]
    
    if filter:
        col = df_p["use_gradient_scaling"]
    else:
        col=None

    fig = px.box(x=x, y=y, color=col, points="outliers",
                 labels={"x": x_name, "y": metric, "color": "Gradient Scaling"},
                 title=f"{metric_title} vs {x_name}")
    if x_name == "Training Sample Size":
        category_order = np.array(sorted(x.unique().astype(int)), dtype=str)
    elif x_name == "name":
        category_order = np.array(sorted(x.unique().astype(str)), dtype=str)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray = category_order)
    return fig

def figure_selection(df: pd.DataFrame, y: str, x: str, title: str, colmap_filter: str):
    df = filter_dataset_colmap(df, colmap_filter)
    fig = px.scatter(df, x=x, y=y, text=None, hover_data=None, title=title)
    selected_points = plotly_events(fig)
    return selected_points

def create_scatterplot(df: pd.DataFrame, y: str, x: str, title:str, colmap_filter: str):
    df = filter_dataset_colmap(df, colmap_filter)
    fig = px.scatter(df, x=x, y=y, text=None, hover_data=None, title=title)
    
    category_order = sorted(df[x])
    fig.update_xaxes(type='category', categoryorder='array', categoryarray = category_order)
    return fig

def select_experiment(df: pd.DataFrame, exp_number: str, only_size: bool = False) -> pd.DataFrame:
    """
    Create selection options for the experiment and return selected experiment
    """
    st.subheader(f"Experiment {exp_number}")
    training_sample_size = st.selectbox(f"Select the training sample size for Experiment {exp_number}", sorted(df["Training Sample Size"].unique()), index=3, key=f"training_sample_size{exp_number}")
    if only_size:
        return df[(df["Training Sample Size"] == training_sample_size)]
    # Convert the selected string "True"/"False" to boolean True/False
    use_gradient_scaling = st.selectbox(f"Select whether gradient scaling was used for Experiment {exp_number}", ["True", "False"], index=0, key=f"use_gradient_scaling{exp_number}")
    use_gradient_scaling = True if use_gradient_scaling == "True" else False

    colmap = st.selectbox("Select the colmap model", ["estimated", "fixed"], index=0,key=f"colmap_filter{exp_number}")

    df_exp = df[(df["Training Sample Size"] == training_sample_size) & (df["use_gradient_scaling"] == use_gradient_scaling)]
    df_exp = filter_dataset_colmap(df_exp, colmap)

    return df_exp

def select_experiment_name(df: pd.DataFrame, exp_number: str, only_size: bool = False) -> pd.DataFrame:
    """
    Create selection options for the experiment and return selected experiment
    """
    st.subheader(f"Experiment {exp_number}")
    experimenty_name = st.selectbox(f"Select the name of the Experiment {exp_number}", sorted(df["name"].unique()), index=0, key=f"experimenty_name{exp_number}")
    df_exp = df[(df["name"] == experimenty_name)]
    if only_size:
        return df_exp
    colmap = st.selectbox("Select the colmap model", ["estimated"], index=0,key=f"colmap_filter{exp_number}")
    df_exp = filter_dataset_colmap(df_exp, colmap)
    return df_exp

def select_stats_alignment_images(experiment_path: str, alignment_measure:str = 'normalized_overlap'):
    experiment_path = Path(experiment_path).parents[3] 
    df = pd.read_csv(experiment_path / "alignment/alignment_scores.csv")
    # Filter out non-finite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df[f"{alignment_measure} Score"].apply(lambda x: pd.Series(x).notna().all())]
    
    # Compute statistics and find corresponding indices
    stats = {'max': df[f"{alignment_measure} Score"].idxmax(), 
             'min': df[f"{alignment_measure} Score"].idxmin(), 
             'median': (df[f"{alignment_measure} Score"] - df[f"{alignment_measure} Score"].median()).abs().idxmin()}
    # Map indices to image names in folder A
    folder_imgs = experiment_path / "images"
    # Read them in the same way they are when computing the allignment - ideally, change allignment to save the name instead of the list index
    img_names = [f.stem for f in folder_imgs.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']]
    selected_imgs = {k: (img_names[int(df.loc[v, 'Image1 Index'])], img_names[int(df.loc[v, 'Image2 Index'])]) for k, v in stats.items()}
    # Formulate corresponding alignment image names for folder B
    folder_align = experiment_path/ "alignment"
    selected_imgs_align = {k: folder_align / f"alignment_{v1}_to_{v2}.png" for k, (v1, v2) in selected_imgs.items()}
    selected_imgs_keypoint = {k: folder_align / f"keypoint_matches_{v1}_to_{v2}.png" for k, (v1, v2) in selected_imgs.items()}
    return selected_imgs_align, selected_imgs_keypoint
     
def get_test_list(gt_images_dir: str) -> Tuple[List[Path], int]:
    """
    Get the list of test images from the ground truth images directory.
    """
    imgs_list = sorted(list(Path(gt_images_dir).glob("images/*.jpg")))
    return imgs_list, len(imgs_list)

def compute_metrics(image_path1: str, image_path2: str) -> float:
    # Load the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Convert the images to tensors and add a batch dimension
    img1 = to_tensor(img1).unsqueeze(0)
    img2 = to_tensor(img2).unsqueeze(0)

    # Ensure the images are in the range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    st.write(f"MSE: {mse.item()}")

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    st.write(f"PSNR: {psnr(img1, img2).item()}")
    st.write(f"SSIM: {ssim(img1, img2).item()}")
    st.write(f"LPIPS: {lpips(img1, img2).item()}")
    
def read_alignemnt_metrics_file(df: pd.DataFrame, metric: str="normalized_overlap"):
    experiments_dict = {}
    for idx in range(len(df)):
        experiment_path = Path(df["output_path_nerf"].iloc[idx]).parents[3] / "alignment/alignment_scores.csv"
        experiments_dict[df["name"].iloc[idx]] = pd.read_csv(experiment_path)[f"{metric} Score"].values.tolist()
    experiments_df = pd.DataFrame(list(experiments_dict.items()), columns=['name', metric])
    # experiments_df.set_index('Index', inplace=True)
    return experiments_df
    
def kill_ns_viewer():
    """
    Function to kill the ns-viewer process and any related child processes.
    """
    try:
        subprocess.check_call(["pkill", "-f", "ns-viewer"])
        return True
    except subprocess.CalledProcessError:
        # This means 'pkill' didn't find a process to terminate
        return False