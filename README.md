# Demand Forecasting Streamlit App

Separate, deployable Streamlit application for demand forecasting.

## Setup

1. **Add model file:**
   - Download `demand_advanced_xgb.pkl` from Kaggle
   - Place in `streamlit_app/models/demand_advanced_xgb.pkl`

2. **Add dataset (optional but recommended):**
   - Download `train.csv` from Kaggle
   - Place in `streamlit_app/data/train.csv`

3. **Install dependencies:**
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

4. **Run:**
   ```bash
   streamlit run app.py
   ```

## Deploy

- **Streamlit Cloud:** Push to GitHub, connect to Streamlit Cloud
- **Docker:** Use Dockerfile (if provided)
- **Local:** Just run `streamlit run app.py`

## Structure

```
streamlit_app/
├── app.py                 # Main app
├── requirements.txt       # Dependencies
├── models/                # Model files
│   └── demand_advanced_xgb.pkl
├── data/                  # Dataset (optional)
│   └── train.csv
└── README.md
```
