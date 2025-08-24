import streamlit as st
from process_video import analyze_video
import tempfile
import os
import pandas as pd

st.set_page_config(layout="wide", page_title="Sports AI Analytics", page_icon="⚽")

# --- Initialize Session State ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.results = {}

# --- Custom CSS ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keeping CSS from previous version for brevity

# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/400x200/2B3034/FFFFFF?text=Sports+AI", use_column_width=True)
    st.title("⚙️ Controls")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.subheader("Field Calibration")
    st.info("Measure a known distance on your field_map.png to calculate real-world stats.")
    pixels = st.number_input("Distance in Pixels on Map", value=277)
    meters = st.number_input("Distance in Meters in Real Life", value=16.5)
    
    pixels_per_meter = pixels / meters if meters > 0 else 0

    with st.expander("ℹ️ How It Works"):
        st.write("""This app uses YOLOv8 for detection and Norfair for tracking...""")

# --- Main Page ---
st.title("⚽ Sports AI Analytics Platform")
st.markdown("Upload sports footage to track players, generate heatmaps, and calculate performance statistics.")

uploaded_file = st.file_uploader("Upload Your Sports Video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

if uploaded_file is not None:
    st.video(uploaded_file, start_time=0)

    if st.button("🚀 Analyze Video"):
        st.session_state.analysis_complete = False
        with st.spinner("Analyzing... Please wait."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            video_out, heatmap_out, stats_df = analyze_video(video_path, conf_threshold=confidence_threshold, pixels_per_meter=pixels_per_meter)
            
            st.session_state.results = {
                "video": video_out,
                "heatmap": heatmap_out,
                "stats": stats_df
            }
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("✅ Analysis Complete!")
    
    # --- NEW: Three tabs for results ---
    tab1, tab2, tab3 = st.tabs(["📊 Player Statistics", "📹 Tracked Video", "🔥 Heatmap"])

    with tab1:
        st.header("Performance Metrics")
        st.dataframe(st.session_state.results["stats"], use_container_width=True)

    with tab2:
        st.header("Tracked Video Output")
        st.video(st.session_state.results["video"])
        with open(st.session_state.results["video"], "rb") as file:
            st.download_button("Download Video", file, "tracked_video.avi")

    with tab3:
        st.header("Positional Heatmap")
        st.image(st.session_state.results["heatmap"])
        with open(st.session_state.results["heatmap"], "rb") as file:
            st.download_button("Download Heatmap", file, "heatmap.png")