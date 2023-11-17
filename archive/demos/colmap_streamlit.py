import pandas as pd 
from pathlib import Path
from PIL import Image
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import wandb

# Use @st.cache decorator for functions that take a long time to run and don't want to do it every time. It re-runs it only when the input changes.

# Load data from wandb 
def load_data(entity, project):
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}") 

    data_list, data_list_notfound = [], []
    # Save the data in a dataframe if the state of the run is finished
    for run in runs:
        if run.state == "finished":
            # Check if the run has the summary metrics (i.e. was able to estimate poses)
            if run.summary['Angular Error'] is not None:
                if run.name != "easy-shape-27":
                    output_dir = run.config['output_dir']
                else:
                    output_dir = "/srv/galene0/sr572/palace_of_westminster/dense/output/seed0_nsamples150_I"
                data_list.append({'Name': run.name, 
                                  'Output': output_dir,
                                  'Seed': run.config['seed'],
                                    'Number of Images': round(run.config['sample_size'], 2) if run.config['sample_size'] is not None else None, 
                                    'Rot. Error Mean': round(run.summary['Angular Error']['mean'], 2), 
                                    'Rot. Error Std.': round(run.summary['Angular Error']['std'], 2), 
                                    'Rot. Error Median': round(run.summary['Angular Error']['median'], 2), 
                                    'Rot. Error < 15°': round(run.summary['Angular Error']['percentage_max_dist'], 2), 
                                    'Transl. Error Mean': round(run.summary['L2 Translation Error']['mean'], 2), 
                                    'Transl. Error Std.': round(run.summary['L2 Translation Error']['std'], 2), 
                                    'Transl. Error Median': round(run.summary['L2 Translation Error']['median'], 2), 
                                    'Transl. Error Resc. Mean': round(run.summary['L2 Translation Error']['mean_rescale'], 2), 
                                    'Transl. Error Resc. Std.': round(run.summary['L2 Translation Error']['std_rescale'], 2), 
                                    'Transl. Error Resc. Median': round(run.summary['L2 Translation Error']['median_rescale'], 2), 
                                    'Transl. Error < 25%':round(run.summary['L2 Translation Error']['percentage_max_dist'], 2)})
            else:
                data_list_notfound.append({'Name': run.name, 
                                           'Output': run.config['output_dir'], 
                                           'Seed': run.config['seed'], 
                                           'Number of Images': round(run.config['sample_size'], 2) if run.config['sample_size'] is not None else None, })

    data, data_notfound = pd.DataFrame(data_list), pd.DataFrame(data_list_notfound)
    return data, data_notfound, [i for i in data.columns if "Rot. Error" in i], [i for i in data.columns if "Transl. Error" in i]

# https://plotly.com/python-api-reference/generated/plotly.express.scatter.html

def figure_selection(data, y="Rot. Error Median", x="Number of Images", title="Median Rotation Error vs. Number of Images"):
    fig = px.scatter(data, x=x, y=y, text=None, hover_data=None, title=title)
    selected_points = plotly_events(fig)
    return selected_points

def read_imageslist(filedir):
    #  Load and read undersample_list.txt
    with open(f'{filedir}/undersample_list.txt', 'r') as f:
        undersample_list = f.readlines()
        undersample_list = [i.strip() for i in undersample_list]

    images_path = Path("/srv/galene0/sr572/palace_of_westminster/dense/images")
    images_list = [Image.open(images_path / i) for i in undersample_list]
    return undersample_list, images_list

st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["COLMAP Gold Standard Comparison", "COLMAP Low Sample Size Comparison"])

with tab1:
    st.title("COLMAP Pose Estimation Demo")
    st.markdown("COLMAP was not able to retrieve the poses when using less than **14** images. The following figure shows the results for the runs that were able to estimate the poses.")

    entity, project = "sara", "colmap"  
    data, _, rotational_error_cols, translation_error_cols = load_data(entity, project)

    col1, col2 = st.columns(2)

    selected_points = False

    with col1:
        st.header("Camera Error vs. Number of Images")
        y = st.selectbox("Select the y-axis", rotational_error_cols + translation_error_cols)
        selected_points = figure_selection(data=data, y=y, title=f"{y} vs. Number of Images")

        col1_a, col1_b = st.columns(2)
        if selected_points:
            with col1_a:
                st.subheader("Rotation Error")
                st.dataframe(data.loc[selected_points[0]["pointIndex"]][["Number of Images"] + rotational_error_cols].rename("", inplace=True))
            with col1_b:
                st.subheader("Translation Error")
                st.write(data.loc[selected_points[0]["pointIndex"]][["Number of Images"] + translation_error_cols].rename("", inplace=True))

    with col2:
        st.header("Visualise Reconstruction")
        col2_a, col2_b = st.columns(2)
        with col2_a:
            video_file = open('/srv/galene0/sr572/palace_of_westminster/dense/sparse/reconstruction.mp4', 'rb')
            video_bytes = video_file.read()
            st.subheader("Gold Standard")
            st.video(video_bytes)
        with col2_b:
            if selected_points:
                selected_dir = data.loc[selected_points[0]["pointIndex"]]["Output"]
                video_file = open(f'{selected_dir}/reconstruction.mp4', 'rb')
                video_bytes = video_file.read()
                st.subheader("Estimated")
                st.video(video_bytes)

    with st.expander("Visualise Undersampled Images"):
        if selected_points:
            undersample_list, images_list = read_imageslist(filedir=selected_dir)
            st.image(images_list, caption=undersample_list, width=300)

with tab2:
    with st.sidebar:
        nsamples = st.selectbox("Number of Samples", [5, 7, 10], key="nsamples")
        selection = st.selectbox("Undersampling Process", ["Random", "Non-random"], key="selection")
    
    if nsamples:
        st.title(f"COLMAP Pose Estimation Demo with Low Sample Size - {nsamples} Images")

        wandb_project = f"colmap_lowsamples{nsamples}" if selection == "Random" else f"colmap_lowsamples{nsamples}_selected"
        data_lowsample, data_notfound_lowsample, rotational_error_cols_lowsample,translation_error_cols_lowsample = load_data("sara", wandb_project)

        assert data_lowsample.shape[0]+data_notfound_lowsample.shape[0] == 22, "Number of runs is not 22"

        if selection == "Random":
            text = f"Results of COLMAP using {nsamples} randomly selected images with 22 diffent random seeds. The following figure shows the results for the runs that were able to estimate the poses: {data_lowsample.shape[0]} out of 22."
        elif selection == "Non-random":
            text = f"Results of COLMAP using {nsamples} non-randomly selected images with 22 diffent initialization random seeds. Specifically, given an initial image, the closer $n-1$ images (in terms of translation vectors of the camera pose) are selected. The following figure shows the results for the runs that were able to estimate the poses: {data_lowsample.shape[0]} out of 22."

        st.markdown(text)

        col1_lowsample, col2_lowsample = st.columns(2)

        selected_points_lowsample = False

        with col1_lowsample:
            st.header("Camera Error vs. Number of Images")
            y_lowsample = st.selectbox("Select the y-axis", rotational_error_cols_lowsample + translation_error_cols_lowsample, key="y_lowsample")
            selected_points_lowsample = figure_selection(data=data_lowsample, y=y_lowsample, x="Seed", title=f"{y_lowsample} vs. Number of Images")

            col1_lowsample_a, col1_lowsample_b = st.columns(2)
            if selected_points_lowsample:
                with col1_lowsample_a:
                    st.subheader("Rotation Error")
                    st.dataframe(data_lowsample.loc[selected_points_lowsample[0]["pointIndex"]][["Number of Images"] + rotational_error_cols_lowsample].rename("", inplace=True))
                with col1_lowsample_b:
                    st.subheader("Translation Error")
                    st.write(data_lowsample.loc[selected_points_lowsample[0]["pointIndex"]][["Number of Images"] + translation_error_cols_lowsample].rename("", inplace=True))

        with col2_lowsample:
            if selected_points_lowsample:
                with st.expander("Visualise Undersampled Images"):
                    #  Load and read undersample_list.txt
                    selected_dir_lowsample = data_lowsample.loc[selected_points_lowsample[0]["pointIndex"]]["Output"]
                    undersample_list_lowsample, images_list_lowsample = read_imageslist(filedir=selected_dir_lowsample)
                    st.image(images_list_lowsample, caption=undersample_list_lowsample, width=300)
                    
        if selected_points_lowsample:
            with st.expander(f"Visualise Undersampled Images of Unsuccessful Runs ({data_notfound_lowsample.shape[0]})"):
                seed_list_lowsample_notfound = sorted(data_notfound_lowsample["Seed"].tolist())
                # seed_lowsample_notfound = st.select_slider("Seed", options=seed_list_lowsample_notfound)
                seed_lowsample_notfound = st.selectbox("Seed", options=seed_list_lowsample_notfound)
                
                selected_dir_notfound_lowsample = data_notfound_lowsample[data_notfound_lowsample["Seed"] == seed_lowsample_notfound]["Output"].values[0]
                undersample_list_notfound_lowsample, images_list_notfound_lowsample = read_imageslist(filedir=selected_dir_notfound_lowsample)
                st.image(images_list_notfound_lowsample, caption=undersample_list_notfound_lowsample, width=300)


