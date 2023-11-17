import streamlit as st
from pathlib import Path
from utils import load_wandb_data, load_media, create_plot, select_experiment, get_test_list, compute_metrics
from streamlit_image_comparison import image_comparison


MAXIMUM_CONTENT_WIDTH = 2 * 730

# Streamlit general settings
st.set_page_config(layout="wide")
st.title("NeRF Evaluation")

# Load data
entity, project = "sara", "bridge_of_sighs_colmap_nerf"
df = load_wandb_data(entity, project)

# Two subsections (rows), one after the other
# First Subtitle
st.header("Quantitative Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
# First row
# Create a selection box for the metric. Default is SSIM.
metric = st.selectbox("Select a metric", ["SSIM", "LPIPS", "PSNR"], index=0)
# Create a selection box for using estimated colmap (pose_estimation.colmap_model_path contains processed_data/colmap/sparse/0) or fixed colmap (pose_estimation.colmap_model_path contains processed_data_fixedcolmap/colmap/sparse/0)
colmap = st.selectbox("Select the colmap model", ["estimated", "fixed"], index=0)

st.markdown("Results using the *test set* statistics to center the camera poses to render the new view images.")
fig = create_plot(df, f"{metric} test_scale", metric, colmap)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Results using the *train set* statistics to center the camera poses to render the new view images.")
fig = create_plot(df, f"{metric} train_scale", metric, colmap)
st.plotly_chart(fig, use_container_width=True)


# Second row
# Second Subtitle
st.markdown("---")  # horizontal line
st.header("Qualitative Evaluation")
# Second row
col1, col2 = st.columns(2)

with col1:
    df_exp1 = select_experiment(df, '1')
    with st.expander("Rendered Video 1", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp1["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.subheader(f"Experiment name: {df_exp1.name.iloc[0]}")
        st.video(video_bytes)

with col2:
    df_exp2 = select_experiment(df, '2')
    with st.expander("Rendered Video 2", expanded=True):
        # Load the rendered video
        video_path = Path(df_exp2["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
        video_bytes = load_media(video_path, 'video')
        st.subheader(f"Experiment name: {df_exp2.name.iloc[0]}")
        st.video(video_bytes)

# Create a slider for the test sample.
gt_images_dir = Path(df["gt_images_dir"].unique()[0])
test_list, test_size = get_test_list(gt_images_dir)
st.markdown("<h5 style='text-align:center;'>Select the test sample to compare</h5>", unsafe_allow_html=True)
idx = st.select_slider(label='', options=list(range(1, test_size + 1)), label_visibility='collapsed', )
# gt_image_path = gt_images_dir / "images" / test_list[idx-1].name
gt_image_path = Path("/workspace/data/bridge_of_sighs/output/gold_standard/processed_data/images_2") / test_list[idx-1].name


with col1:
    # Create a select box for the statistics used to center the camera pose at evaluation time
    center_statistics = st.selectbox("Select the statistics used to center the camera pose", ["test", "train"], index=0, key='center_stats1')

    # Load the rendered image
    rendered_image_path = Path(df_exp1["output_path_nerf"].iloc[0]) / f"evaluation/{center_statistics}_scale/images" / test_list[idx-1].name
    rendered_image = load_media(rendered_image_path, 'image')
    st.subheader(f"Rendered Images 1")
    # : {test_list[idx-1].name}
    # st.image(rendered_image, use_column_width=True)

    image_comparison(
    img1=str(rendered_image_path),
    img2=str(gt_image_path),
    label1="Rendered",
    label2="GT",
    # width=700,
    # use_column_width=True,
    starting_position=50,
    show_labels=True,
    # make_responsive=True,
    # in_memory=True,
    )
    compute_metrics(str(rendered_image_path), str(gt_image_path))

with col2:
    # Create a select box for the statistics used to center the camera pose at evaluation time
    center_statistics = st.selectbox("Select the statistics used to center the camera pose", ["test", "train"], index=0, key='center_stats2')

    # Load the rendered image
    rendered_image_path = Path(df_exp2["output_path_nerf"].iloc[0]) / f"evaluation/{center_statistics}_scale/images" / test_list[idx-1].name

    rendered_image = load_media(rendered_image_path, 'image')
    st.subheader("Rendered Images 2")
    # st.image(rendered_image, use_column_width=True)

    image_comparison(
    img1=str(rendered_image_path),
    img2=str(gt_image_path),
    label1="Rendered",
    label2="GT",
    # width=700,
    # use_column_width=True,
    starting_position=50,
    show_labels=True,
    # make_responsive=True,
    # in_memory=True,
    )
    compute_metrics(str(rendered_image_path), str(gt_image_path))

st.markdown("""
    <style>
    iframe {
        width: 100%;
        max-height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# # Third Subtitle
# st.markdown("---")  # horizontal line
# st.header("Open NeRF Studio Viewer")
# # Third row
# df_exp_viewer = select_experiment(df, '')["output_config_path"].iloc[0]

# ns-viewer --load-config {outputs/.../config.yml}