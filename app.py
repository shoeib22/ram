import streamlit as st
from process_video import analyze_video
import tempfile
import os


st.set_page_config(
    layout="wide",
    page_title="Sports AI Analytics",
    page_icon="⚽"
)


if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.video_out_path = None
    st.session_state.heatmap_out_path = None


st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Button styling */
    .stButton>button {
        border: 2px solid #4CAF50;
        border-radius: 20px;
        padding: 10px 20px;
        background-color: transparent;
        color: #4CAF50;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    /* Card styling */
    .card {
        background-color: #2B3034;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .card-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.image("https://placehold.co/400x200/2B3034/FFFFFF?text=Sports+AI", use_column_width=True)
    st.title("⚙️ Controls")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.info("Adjust the model's sensitivity. Higher values are stricter.")
    
    with st.expander("ℹ️ How It Works"):
        st.write("""
            This app uses the YOLOv8 model for object detection and the Norfair library for tracking. A perspective transformation is applied to map player positions onto a 2D field to generate a heatmap.
        """)


st.title("⚽ Sports AI Analytics Platform")
st.markdown("Upload sports footage to automatically track players and generate positional heatmaps.")

uploaded_file = st.file_uploader("Upload Your Sports Video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

if uploaded_file is not None:
    # Display a smaller preview of the uploaded video
    st.video(uploaded_file, start_time=0)

    if st.button("🚀 Analyze Video"):
        st.session_state.analysis_complete = False # Reset on new analysis
        with st.spinner("Analyzing... Please wait. This can take several minutes."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            video_out, heatmap_out = analyze_video(video_path, conf_threshold=confidence_threshold)
            
            # Store results in session state
            st.session_state.video_out_path = video_out
            st.session_state.heatmap_out_path = heatmap_out
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("✅ Analysis Complete!")
    
    st.markdown("---")
    st.markdown('<p class="card-header">📊 Results</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Tracked Video Output")
        st.video(st.session_state.video_out_path)
        with open(st.session_state.video_out_path, "rb") as file:
            st.download_button("Download Video", file, "tracked_video.avi")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Positional Heatmap")
        st.image(st.session_state.heatmap_out_path)
        with open(st.session_state.heatmap_out_path, "rb") as file:
            st.download_button("Download Heatmap", file, "heatmap.png")
        st.markdown('</div>', unsafe_allow_html=True)