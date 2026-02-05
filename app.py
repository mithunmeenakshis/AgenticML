import streamlit as st
import pandas as pd
import json
import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import time

# Configure page
st.set_page_config(
    page_title="AgenticML - AI-Powered AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .feature-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .success-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .error-card {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def create_user_session_folder(username):
    """Create a folder for user with today's date"""
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"sessions/{username}_{today}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    
    # Copy necessary files to session folder
    files_to_copy = ['101.py', 'models.py', 'api_key.txt']
    for file_name in files_to_copy:
        source_path = Path(file_name)
        if source_path.exists():
            destination_path = Path(folder_name) / file_name
            shutil.copy2(source_path, destination_path)
    
    return folder_name

def get_existing_sessions():
    """Get all existing session folders"""
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for folder in sessions_dir.iterdir():
        if folder.is_dir():
            # Extract username and date from folder name
            parts = folder.name.split('_')
            if len(parts) >= 2:
                username = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                date_time = '_'.join(parts[-2:])
                
                # Check if report exists
                report_path = folder / "report.json"
                if report_path.exists():
                    sessions.append({
                        'folder': folder.name,
                        'username': username,
                        'datetime': date_time,
                        'path': str(folder)
                    })
    
    return sorted(sessions, key=lambda x: x['datetime'], reverse=True)

def load_report(session_path):
    """Load report from session folder"""
    report_path = Path(session_path) / "report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            return json.load(f)
    return None

def create_download_zip(session_path):
    """Create a zip file with all session files"""
    session_folder = Path(session_path)
    zip_path = session_folder / "download_package.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in session_folder.iterdir():
            if file.is_file() and file.name != "download_package.zip":
                zipf.write(file, file.name)
    
    return zip_path

def display_feature_engineering_results(report):
    """Display feature engineering results in a nice format"""
    st.subheader("🔧 Feature Engineering Results")
    
    engineered_features = report.get('engineered_features', [])
    successful_features = [f for f in engineered_features if 'success' in f['status']]
    failed_features = [f for f in engineered_features if 'failed' in f['status']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("✅ Successful Features", len(successful_features))
    with col2:
        st.metric("❌ Failed Features", len(failed_features))
    
    # Show successful features
    if successful_features:
        st.markdown("### ✅ Successfully Created Features")
        for feature in successful_features:
            st.markdown(f"""
            <div class="success-card">
                <strong>{feature['feature']}</strong><br>
                <em>{feature['description']}</em><br>
                <code>{feature['code']}</code>
            </div>
            """, unsafe_allow_html=True)
    
    # Show failed features
    if failed_features:
        with st.expander("❌ Failed Feature Attempts", expanded=False):
            for feature in failed_features:
                st.markdown(f"""
                <div class="error-card">
                    <strong>{feature['feature']}</strong><br>
                    <em>{feature['description']}</em><br>
                    <code>{feature['code']}</code><br>
                    <small>Error: {feature['status']}</small>
                </div>
                """, unsafe_allow_html=True)

def display_model_results(report):
    """Display model performance results"""
    st.subheader("🎯 Model Performance")
    
    model_results = report.get('model_results', [])
    best_model = report.get('best_model', {})
    
    if model_results:
        # Create performance chart
        models = []
        scores = []
        for result in model_results:
            models.append(result['model'])
            scores.append(result['score'])
        
        fig = px.bar(
            x=models, 
            y=scores,
            title="Model Performance Comparison",
            labels={'x': 'Models', 'y': 'Score'},
            color=scores,
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        if best_model:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model: {best_model['name'].upper()}</h3>
                <h2>{best_model['score']:.4f}</h2>
                <p>Final Score</p>
            </div>
            """, unsafe_allow_html=True)

def display_data_overview(report):
    """Display data overview"""
    st.subheader("📊 Data Overview")
    
    data_overview = report.get('data_overview', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Rows", data_overview.get('num_rows', 'N/A'))
    with col2:
        st.metric("📊 Columns", data_overview.get('num_columns', 'N/A'))
    with col3:
        numerical_cols = len(data_overview.get('feature_types', {}).get('numerical', []))
        st.metric("🔢 Numerical", numerical_cols)
    with col4:
        categorical_cols = len(data_overview.get('feature_types', {}).get('categorical', []))
        st.metric("📝 Categorical", categorical_cols)
    
    # Missing values chart
    missing_values = data_overview.get('missing_values', {})
    if missing_values:
        missing_df = pd.DataFrame.from_dict(missing_values, orient='index', columns=['Missing Count'])
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                y=missing_df.index,
                x='Missing Count',
                orientation='h',
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 AgenticML</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Automated Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🆕 New Analysis", "📂 View Previous Runs"]
    )
    
    if mode == "🆕 New Analysis":
        st.header("🆕 Start New Analysis")
        
        # User input
        col1, col2 = st.columns([2, 1])
        with col1:
            username = st.text_input("👤 Enter your name:", placeholder="e.g., John Doe")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload your CSV dataset",
            type=['csv'],
            help="Upload a CSV file for automated machine learning analysis"
        )
        
        if uploaded_file and username:
            # Display file info
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Target column selection
            target_column = st.selectbox(
                "🎯 Select target column (what you want to predict):",
                options=df.columns.tolist(),
                help="Choose the column you want to predict"
            )
            
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                # Create session folder
                session_folder = create_user_session_folder(username)
                
                # Save uploaded file
                csv_path = Path(session_folder) / "data.csv"
                df.to_csv(csv_path, index=False)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Initializing AI analysis...")
                    progress_bar.progress(10)
                    
                    # Run the ML pipeline
                    result = subprocess.run([
                        "python", "101.py",
                        "--csv", "data.csv",  # Use relative path since we're in session folder
                        "--target", target_column
                    ], 
                    cwd=session_folder,
                    capture_output=True, 
                    text=True,
                    timeout=600  # 10 minute timeout
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("📊 Generating report...")
                    
                    if result.returncode == 0:
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis completed successfully!")
                        
                        st.success("🎉 Analysis completed! Results are ready.")
                        
                        # Load and display results
                        report = load_report(session_folder)
                        if report:
                            st.balloons()
                            
                            # Display results
                            display_data_overview(report)
                            display_feature_engineering_results(report)
                            display_model_results(report)
                            
                            # Download section
                            st.header("📥 Download Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Report download
                                report_path = Path(session_folder) / "report.json"
                                if report_path.exists():
                                    with open(report_path, 'r') as f:
                                        report_data = f.read()
                                    st.download_button(
                                        "📄 Download Report (JSON)",
                                        data=report_data,
                                        file_name=f"{username}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
                            
                            with col2:
                                # Model download
                                model_path = Path(session_folder) / "best_model.pkl"
                                if model_path.exists():
                                    with open(model_path, 'rb') as f:
                                        model_data = f.read()
                                    st.download_button(
                                        "🤖 Download Model (PKL)",
                                        data=model_data,
                                        file_name=f"{username}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                        mime="application/octet-stream"
                                    )
                            
                            # Complete package download
                            zip_path = create_download_zip(session_folder)
                            with open(zip_path, 'rb') as f:
                                zip_data = f.read()
                            st.download_button(
                                "📦 Download Complete Package (ZIP)",
                                data=zip_data,
                                file_name=f"{username}_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                            
                        else:
                            st.error("❌ Could not load report file.")
                    else:
                        progress_bar.progress(0)
                        status_text.text("❌ Analysis failed.")
                        st.error("❌ Analysis failed. Please check your data and try again.")
                        st.code(result.stderr)
                        
                except subprocess.TimeoutExpired:
                    st.error("⏰ Analysis timed out. Please try with a smaller dataset.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
    
    else:  # View Previous Runs
        st.header("📂 Previous Analysis Sessions")
        
        sessions = get_existing_sessions()
        
        if not sessions:
            st.info("📭 No previous sessions found. Start a new analysis to see results here!")
            return
        
        # Session selector
        session_options = [
            f"👤 {s['username']} - 📅 {s['datetime']}" for s in sessions
        ]
        
        selected_session_idx = st.selectbox(
            "🔍 Select a session to view:",
            range(len(session_options)),
            format_func=lambda x: session_options[x]
        )
        
        if selected_session_idx is not None:
            selected_session = sessions[selected_session_idx]
            session_path = selected_session['path']
            
            # Load and display report
            report = load_report(session_path)
            
            if report:
                st.success(f"✅ Loaded session for {selected_session['username']}")
                
                # Display session info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👤 User", selected_session['username'])
                with col2:
                    st.metric("📅 Date", selected_session['datetime'].split('_')[0])
                with col3:
                    st.metric("⏰ Time", selected_session['datetime'].split('_')[1].replace('-', ':'))
                
                # Display results
                display_data_overview(report)
                display_feature_engineering_results(report)
                display_model_results(report)
                
                # Download section for previous sessions
                st.header("📥 Download Previous Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Report download
                    report_path = Path(session_path) / "report.json"
                    if report_path.exists():
                        with open(report_path, 'r') as f:
                            report_data = f.read()
                        st.download_button(
                            "📄 Download Report (JSON)",
                            data=report_data,
                            file_name=f"{selected_session['username']}_report_{selected_session['datetime']}.json",
                            mime="application/json"
                        )
                
                with col2:
                    # Model download
                    model_path = Path(session_path) / "best_model.pkl"
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            model_data = f.read()
                        st.download_button(
                            "🤖 Download Model (PKL)",
                            data=model_data,
                            file_name=f"{selected_session['username']}_model_{selected_session['datetime']}.pkl",
                            mime="application/octet-stream"
                        )
                
                # Complete package download
                zip_path = create_download_zip(session_path)
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                st.download_button(
                    "📦 Download Complete Package (ZIP)",
                    data=zip_data,
                    file_name=f"{selected_session['username']}_complete_{selected_session['datetime']}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.error("❌ Could not load report for this session.")

if __name__ == "__main__":
    main()
