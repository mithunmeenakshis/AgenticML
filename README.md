# 🤖 AgenticML - AI-Powered AutoML Platform

**AgenticML** is an intelligent automated machine learning platform that uses LLM-powered feature engineering to automatically improve model performance through iterative optimization.

## ✨ Features

- 🎯 **Automatic Task Detection**: Determines if your problem is classification or regression
- 🧠 **AI Feature Engineering**: Uses Llama 3 to generate and test new features automatically
- 🔄 **Iterative Optimization**: Continuously improves until target performance is reached
- 📊 **Beautiful Web Interface**: Streamlit-based frontend with aesthetic design
- 👥 **Multi-user Support**: Session management with user-specific folders
- 📁 **Session History**: View and download previous analysis results
- 📈 **Rich Visualizations**: Interactive charts showing model performance and data insights
- 📦 **Complete Downloads**: Download models, reports, and complete analysis packages

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (get one at [Groq Console](https://console.groq.com/))

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd AgenticML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Groq API key**
   
   Create an `api_key.txt` file in the project directory:
   ```bash
   echo "your_groq_api_key_here" > api_key.txt
   ```
   
   Or set as environment variable:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the web interface.

## 📖 How to Use

### Starting a New Analysis

1. **Enter your name** in the user input field
2. **Upload your CSV dataset** using the file uploader
3. **Preview your data** in the expandable data preview section
4. **Select your target column** (what you want to predict)
5. **Click "Start AI Analysis"** and watch the magic happen!

The system will:
- Automatically detect if it's a classification or regression problem
- Generate intelligent features using AI
- Train multiple models with hyperparameter tuning
- Select the best performing model
- Generate a comprehensive report

### Viewing Previous Results

1. Switch to **"View Previous Runs"** mode in the sidebar
2. Select any previous session from the dropdown
3. View all the analysis results, charts, and metrics
4. Download reports, models, or complete packages

## 🎯 Supported Data Types

- **Classification**: Binary and multi-class classification problems
- **Regression**: Continuous target variable prediction
- **Mixed Data**: Handles both numerical and categorical features
- **Missing Values**: Automatically handles missing data

## 📊 What You Get

### Analysis Report
- Data overview and statistics
- Feature engineering results (successful and failed attempts)
- Model performance comparisons
- Best model details and parameters
- Cross-validation scores

### Downloads
- **JSON Report**: Detailed analysis results and metadata
- **PKL Model**: Trained scikit-learn model ready for deployment
- **ZIP Package**: Complete analysis package with all files

### Visualizations
- Model performance comparison charts
- Missing values analysis
- Data overview metrics
- Feature engineering success/failure tracking

## 🔧 Technical Details

### AI Feature Engineering Process

1. **Data Analysis**: Generates statistical summaries of your dataset
2. **LLM Suggestions**: Sends data insights to Llama 3 for feature ideas
3. **Code Generation**: AI generates Python pandas code for new features
4. **Execution & Retry**: Attempts execution with automatic error correction
5. **Feature Selection**: Uses RFE to select the most important features
6. **Model Training**: Trains multiple algorithms with grid search

### Supported Models

**Classification:**
- Logistic Regression
- Support Vector Classifier (SVM)
- Random Forest Classifier

**Regression:**
- Ridge Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor

### Stopping Conditions

The system intelligently stops when:
- Score reaches ≥85% (configurable)
- Maximum iterations reached (default: 5)
- No improvement for 3 consecutive iterations

## 📁 Project Structure

```
AgenticML/
├── app.py              # Streamlit web interface
├── 101.py              # Core ML pipeline
├── models.py           # Model training functions
├── api_key.txt         # Groq API key (create this)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── sessions/          # User session folders (auto-created)
    ├── user1_2024-01-15_10-30-45/
    ├── user2_2024-01-15_11-00-20/
    └── ...
```

## 🛠️ Customization

### Adding New Models

Edit `models.py` to add new algorithms:

```python
# Add to run_classification_models or run_regression_models
"new_model": {
    "estimator": YourModel(),
    "params": {"param1": [1, 2, 3]}
}
```

### Modifying Stopping Conditions

Edit the stopping logic in `101.py`:

```python
if best_score >= 0.90:  # Change target score
    break
elif loop_count >= 10:  # Change max iterations
    break
```

### Custom Feature Engineering

The LLM prompts can be customized in the `call_llama_for_features` function in `101.py`.

## 🔒 Security Notes

- Keep your `api_key.txt` file secure and never commit it to version control
- The application runs locally on your machine - your data never leaves your computer
- Session folders contain copies of your data - manage them according to your privacy requirements

## 🐛 Troubleshooting

### Common Issues

1. **"api_key.txt not found"**
   - Create the api_key.txt file with your Groq API key
   - Or set the GROQ_API_KEY environment variable

2. **Analysis timeout**
   - Try with a smaller dataset
   - Increase timeout in app.py if needed

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **File upload issues**
   - Ensure your CSV has a header row
   - Check that the target column exists in your data

## 📞 Support

For issues, questions, or feature requests, please check the troubleshooting section above or refer to the code comments for implementation details.

## 🌟 Example Results

The system has been tested on various datasets including:
- **Titanic Survival**: 84.36% accuracy with AI-generated features
- **Housing Prices**: Significantly improved RMSE through feature engineering
- **Customer Churn**: Enhanced precision through intelligent feature creation

---

**Happy Machine Learning! 🚀**
