# Source Code Directory

This directory contains the production-ready source code for the Thyroid Disease Likelihood Streamlit web application and utility modules.

## Directory Structure

```
src/
├── app.py                      # Main Streamlit application
└── utils/                      # Utility modules
    ├── __init__.py             # Package initialization
    ├── data_processor.py       # Data preprocessing and validation
    ├── model_loader.py         # Model loading and management
    ├── predictor.py            # Prediction engine
    └── ai_insights.py          # OpenAI GPT integration
```

## Main Application

### `app.py`

The main Streamlit web application providing an interactive interface for thyroid disease classification.

#### Features

- Project overview and introduction
- Key metrics and model performance
- Navigation guidance
- Interactive data-entry form with validation
- Binary and multiclass predictions
- AI-generated insights and recommendation (powered by GPT-4)

#### Technical Stack
- **Framework**: Streamlit
- **AI Integration**: OpenAI GPT-4 API
- **State Management**: Streamlit session state

#### Running the Application

```bash
# From project root
streamlit run src/app.py

# Or from src directory
cd src
streamlit run app.py
```

#### Configuration

Environment variables (create `.env` file):
```bash
OPENAI_API_KEY=your_api_key_here
```

## Utility Modules

### `utils/data_processor.py`

Handles all data preprocessing, validation, and transformation.


### `utils/model_loader.py`

Manages model loading, caching, and version control.

### `utils/predictor.py`

Core prediction engine for both binary and multiclass classification.

### `utils/ai_insights.py`

OpenAI GPT-4 integration for natural language explanations and recommendations.


## Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# Run application
streamlit run src/app.py
```

