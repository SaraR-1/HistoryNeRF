import streamlit as st
import subprocess
from pathlib import Path
from utils import load_wandb_data, load_media, create_plot, select_experiment, get_test_list, compute_metrics, kill_ns_viewer
from streamlit_image_comparison import image_comparison

MAXIMUM_CONTENT_WIDTH = 2 * 730

# Streamlit general settings
st.set_page_config(layout="wide")
st.title("NeRF Evaluation")

# Load data
entity, project = "sara", "old_petrol_pumps"
df = load_wandb_data(entity, project)

# Two subsections (rows), one after the other
# First Subtitle
st.header("Quantitative Evaluation")
st.markdown("<br>", unsafe_allow_html=True)  # line break
# First row
# Create a selection box for the metric. Default is SSIM.
metric = st.selectbox("Select a metric", ["SSIM", "LPIPS", "PSNR"], index=0)

# st.markdown("Results using the *test set* statistics to center the camera poses to render the new view images.")
# fig = create_plot(df, f"{metric} test_scale", metric, "estimated", filter=False)
# st.plotly_chart(fig, use_container_width=True)

# st.markdown("Results using the *train set* statistics to center the camera poses to render the new view images.")
st.markdown(f"Distribution of the pair-wise {metric} metric. For each of the 80 images we compare the real image and the one generated using NeRF.")
fig = create_plot(df, f"{metric} train_scale", metric, "estimated", filter=False)
st.plotly_chart(fig, use_container_width=True)

# Second row
# Second Subtitle
# st.markdown("---")  # horizontal line
# st.header("Qualitative Evaluation")
# Second row
# col1, col2 = st.columns(2)

# with col1:
#     df_exp1 = select_experiment(df, '1', True)
#     with st.expander("Rendered Video 1", expanded=True):
#         # Load the rendered video
#         video_path = Path(df_exp1["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
#         video_bytes = load_media(video_path, 'video')
#         st.subheader(f"Experiment name: {df_exp1.name.iloc[0]}")
#         st.video(video_bytes)

# with col2:
#     df_exp2 = select_experiment(df, '2', True)
#     with st.expander("Rendered Video 2", expanded=True):
#         # Load the rendered video
#         video_path = Path(df_exp2["output_path_nerf"].iloc[0]) / "evaluation/output.mp4"
#         video_bytes = load_media(video_path, 'video')
#         st.subheader(f"Experiment name: {df_exp2.name.iloc[0]}")
#         st.video(video_bytes)
        
        
# Third Subtitle
st.markdown("---")  # horizontal line
st.header("Qualitative Evaluation - Open NeRF Studio Viewer")
st.markdown("Evalaute the 3D model generated using NeRF.")
# Third row
df_exp_viewer = select_experiment(df, '', True)["output_config_path"].iloc[0]

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
