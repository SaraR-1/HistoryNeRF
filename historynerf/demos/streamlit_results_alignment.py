import streamlit as st
from pathlib import Path
from utils import load_wandb_data, create_plot, read_alignemnt_metrics_file, load_media, select_experiment_name, select_stats_alignment_images, create_scatterplot, figure_selection, select_experiment, get_test_list, compute_metrics
from streamlit_image_comparison import image_comparison


MAXIMUM_CONTENT_WIDTH = 2 * 730

# Streamlit general settings
st.set_page_config(layout="wide")
st.title("NeRF Evaluation")

# Load data
entity, project = "sara", "bridge_of_sighs_colmap_nerf_alignment"
df = load_wandb_data(entity, project)

st.header("Quantitative Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
metric = st.selectbox("Select a metric", ["SSIM", "LPIPS", "PSNR"], index=0)

st.markdown("Results using the *train set* statistics to center the camera poses to render the new view images.")
fig = create_plot(df, f"{metric} train_scale", metric, colmap_filter="estimated", x_name="name")
st.plotly_chart(fig, use_container_width=True)


st.header("Alignment Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
alignment_metric = st.selectbox("Select an alignment measure", ["normalized_overlap", "ssd", "mse", "ncc", "refined_normalized_overlap"], index=0)

df_alignment = read_alignemnt_metrics_file(df, alignment_metric)
st.markdown("Explain what's here")
fig = create_plot(df_alignment, alignment_metric, alignment_metric, colmap_filter="estimated", x_name="name", filter=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")  # horizontal line
st.header("Qualitative Evaluation")
# Second row
col1, col2 = st.columns(2)

with col1:
    df_exp1 = select_experiment_name(df, '1')
    with st.expander("Rendered Video 1", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp1["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.subheader(f"Experiment name: {df_exp1.name.iloc[0]}")
        st.video(video_bytes)
    with st.expander("Alignment Examples", expanded=True):
        examples_name_align, examples_name_keypoint = select_stats_alignment_images(Path(df_exp1["output_path_nerf"].iloc[0]))
        for stat in ["max", "median", "min"]:
            st.subheader(f"Allignment Example {stat}")
            image = load_media(examples_name_align[stat], 'image')
            st.image(image)
    with st.expander("KeyPoints Match Examples", expanded=True):
        for stat in ["max", "median", "min"]:
            st.subheader(f"KeyPoints Match Example {stat}")
            image = load_media(examples_name_keypoint[stat], 'image')
            st.image(image)
        

with col2:
    df_exp2 = select_experiment_name(df, '2')
    with st.expander("Rendered Video 2", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp2["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.subheader(f"Experiment name: {df_exp2.name.iloc[0]}")
        st.video(video_bytes)
    with st.expander("Alignment Examples", expanded=True):
        examples_name_align, examples_name_keypoint = select_stats_alignment_images(Path(df_exp2["output_path_nerf"].iloc[0]))
        for stat in ["max", "median", "min"]:
            st.subheader(f"Allignment Example {stat}")
            image = load_media(examples_name_align[stat], 'image')
            st.image(image)
    with st.expander("KeyPoints Match Examples", expanded=True):
        for stat in ["max", "median", "min"]:
            st.subheader(f"KeyPoints Match Example {stat}")
            image = load_media(examples_name_keypoint[stat], 'image')
            st.image(image)
        
        