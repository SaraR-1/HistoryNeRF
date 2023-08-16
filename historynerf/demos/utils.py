from typing import Union, List, Tuple
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import wandb
import numpy as np
import plotly.express as px
import streamlit as st

import torch
from torchvision.transforms.functional import to_tensor
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load_wandb_data(entity: str, project: str) -> pd.DataFrame:
    '''
    Load the data from W&B and return a dataframe.
    '''
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}") 

    data = []
    for run in runs:
        if run.state == "finished":
            try:
                data.append({
                    "name": run.name,
                    "use_gradient_scaling": run.config["nerf"]["pipeline"]["model"]["use_gradient_scaling"], 
                    "max_num_iterations": run.config["nerf"]["max_num_iterations"],
                    "method_name": run.config["nerf"]["method_name"],
                    "output_path_nerf": run.config["output_path_nerf"],
                    "colmap_model_path": run.config["pose_estimation"]["colmap_model_path"] if run.config["pose_estimation"]["colmap_model_path"] is not None else "",
                    "gt_images_dir": run.config["evaluation"]["gt_images_dir"],
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

def create_plot(df: pd.DataFrame, metric: str, metric_title:str, colmap_filter: str):
    df = filter_dataset_colmap(df, colmap_filter)
    df_p = df.explode(metric)
    x = df_p["Training Sample Size"].astype(str)
    y = df_p[metric]
    col = df_p["use_gradient_scaling"]

    fig = px.box(x=x, y=y, color=col, points="outliers",
                 labels={"x": "Training Sample Size", "y": metric, "color": "Gradient Scaling"},
                 title=f"{metric_title} vs Training Sample Size")

    category_order = np.array(sorted(x.unique().astype(int)), dtype=str)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray = category_order)
    return fig

def select_experiment(df: pd.DataFrame, exp_number: str) -> pd.DataFrame:
    """
    Create selection options for the experiment and return selected experiment
    """
    st.subheader(f"Experiment {exp_number}")
    training_sample_size = st.selectbox(f"Select the training sample size for Experiment {exp_number}", sorted(df["Training Sample Size"].unique()), index=3, key=f"training_sample_size{exp_number}")
    # Convert the selected string "True"/"False" to boolean True/False
    use_gradient_scaling = st.selectbox(f"Select whether gradient scaling was used for Experiment {exp_number}", ["True", "False"], index=0, key=f"use_gradient_scaling{exp_number}")
    use_gradient_scaling = True if use_gradient_scaling == "True" else False

    colmap = st.selectbox("Select the colmap model", ["estimated", "fixed"], index=0,key=f"colmap_filter{exp_number}")

    df_exp = df[(df["Training Sample Size"] == training_sample_size) & (df["use_gradient_scaling"] == use_gradient_scaling)]
    df_exp = filter_dataset_colmap(df_exp, colmap)

    return df_exp


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
    


