import streamlit as st
import subprocess
from pathlib import Path
from utils import load_wandb_data, select_experiment_name, load_media, create_plot, select_experiment, get_test_list, compute_metrics, kill_ns_viewer
from streamlit_image_comparison import image_comparison

MAXIMUM_CONTENT_WIDTH = 2 * 730

# Streamlit general settings
st.set_page_config(layout="wide")
st.title("NeRF Evaluation")

# Load data
entity, project = "sara", "dino"
df = load_wandb_data(entity, project, ["finished", "killed"])
df["Angle"] = df["output_path_nerf"].str.extract(r'updown(\d+)')
st.table(df[["name", "Angle", "Training Sample Size"]])

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
