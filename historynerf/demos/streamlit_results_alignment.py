import streamlit as st
from pathlib import Path
import subprocess
from utils import load_wandb_data, create_plot, read_alignemnt_metrics_file, load_media, select_experiment_name, select_stats_alignment_images, create_scatterplot, figure_selection, select_experiment, get_test_list, compute_metrics, kill_ns_viewer
from streamlit_image_comparison import image_comparison


MAXIMUM_CONTENT_WIDTH = 2 * 730

# Streamlit general settings
st.set_page_config(layout="wide")
st.title("NeRF Evaluation")
st.markdown("The objective of this study is to investigate variations in NeRF performance based on distinct training subsets comprising 30 images. These subsets are either randomly sampled from the entire training corpus or consist of a contiguous sequence of 30 images.")
# Load data
entity, project = "sara", "bridge_of_sighs_colmap_nerf_alignment"
df = load_wandb_data(entity, project)

st.header("Quantitative Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
metric = st.selectbox("Select a metric to compare the original images and the NeRF-generated renderings:", ["SSIM", "LPIPS", "PSNR"], index=0)

st.markdown("The evaluation employs train set statistics to calibrate camera poses for rendering the *new view* images. The chosen metric assesses a subset of 80 previously unseen views.")
fig = create_plot(df, f"{metric} train_scale", metric, colmap_filter="estimated", x_name="name", filter=False)
st.plotly_chart(fig, use_container_width=True)


st.header("Alignment Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
alignment_metric = st.selectbox("Select an alignment measure:", ["normalized_overlap", "ssd", "mse", "ncc", "refined_normalized_overlap"], index=0)

df_alignment = read_alignemnt_metrics_file(df, alignment_metric)
st.markdown("The findings present the degree of alignment (overlap) between each image pair within the training set. The plot depicts the distribution of this measure across all possible pairings.")
fig = create_plot(df_alignment, alignment_metric, alignment_metric, colmap_filter="estimated", x_name="name", filter=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")  # horizontal line
st.header("Qualitative Evaluation")
# Second row
col1, col2 = st.columns(2)

with col1:
    df_exp1 = select_experiment_name(df, '1', True)
    with st.expander("Rendered Video 1", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp1["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.markdown(f"**Experiment name: {df_exp1.name.iloc[0]}**")
        st.video(video_bytes)
    with st.expander("Alignment Examples", expanded=True):
        examples_name_align, examples_name_keypoint = select_stats_alignment_images(Path(df_exp1["output_path_nerf"].iloc[0]))
        for stat in ["max", "median", "min"]:
            st.markdown(f"**Alignment example of the pair with the {stat} alignment value across all possible combinations.**")
            image = load_media(examples_name_align[stat], 'image')
            st.image(image)
    with st.expander("KeyPoints Match Examples", expanded=True):
        for stat in ["max", "median", "min"]:
            st.markdown(f"**KeyPoints match example of the pair with the {stat} alignment value across all possible combinations.**")
            image = load_media(examples_name_keypoint[stat], 'image')
            st.image(image)
        

with col2:
    df_exp2 = select_experiment_name(df, '2', True)
    with st.expander("Rendered Video 2", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp2["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.markdown(f"**Experiment name: {df_exp2.name.iloc[0]}**")
        st.video(video_bytes)
    with st.expander("Alignment Examples", expanded=True):
        examples_name_align, examples_name_keypoint = select_stats_alignment_images(Path(df_exp2["output_path_nerf"].iloc[0]))
        for stat in ["max", "median", "min"]:
            st.markdown(f"**Alignment example of the pair with the {stat} alignment value across all possible combinations.**")
            image = load_media(examples_name_align[stat], 'image')
            st.image(image)
    with st.expander("KeyPoints Match Examples", expanded=True):
        for stat in ["max", "median", "min"]:
            st.markdown(f"**KeyPoints match example of the pair with the {stat} alignment value across all possible combinations.**")
            image = load_media(examples_name_keypoint[stat], 'image')
            st.image(image)
        
 # Third Subtitle
st.markdown("---")  # horizontal line
st.header("Open NeRF Studio Viewer")
# Third row
df_exp_viewer = select_experiment_name(df, '', True)["output_config_path"].iloc[0]

ns_viewer_process = None
viewer_url = "https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007"  # Adjust this if the URL is different

    
if st.button('Start ns-viewer'):
    if ns_viewer_process is None or ns_viewer_process.poll() is not None:
        ns_viewer_process = subprocess.Popen(['ns-viewer', '--load-config', df_exp_viewer])
        
        viewer_link = f"[Open ns-viewer]({viewer_url})"
        st.markdown(viewer_link, unsafe_allow_html=True)

        st.write('Started ns-viewer.')
    else:
        st.write('ns-viewer is already running.')

if st.button('Stop ns-viewer'):
    if kill_ns_viewer():
        ns_viewer_process = None
        st.write('Stopped ns-viewer.')
    else:
        st.write('ns-viewer is not running or couldnt be terminated.')
       