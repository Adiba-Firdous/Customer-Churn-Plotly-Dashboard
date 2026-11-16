# Customer Churn Model Evaluation Dashboard

An interactive Streamlit + Plotly dashboard for evaluating a binary customer-churn classification model.

## Features
- Accuracy, precision, recall and F1-score
- Interactive confusion matrix and metric chart
- ROC curve and AUC when probability scores are provided
- Classification report and prediction table
- Upload your own model-predictions CSV
- Adjustable probability threshold

## Project structure
```text
customer_churn_plotly_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_predictions.csv
├── src/
│   └── train_model.py
└── outputs/
    └── .gitkeep
```

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Open Command Prompt / VS Code terminal inside this project folder.
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL printed in the terminal (usually http://localhost:8501).

## Use your previous model's predictions
The dashboard expects a CSV with:
- `y_true`: actual test labels (0 = no churn, 1 = churn)
- `y_pred`: predicted labels (0 or 1)
- `y_probability` (optional): predicted probability of churn, between 0 and 1

Example:
```csv
y_true,y_pred,y_probability
0,0,0.12
1,1,0.87
1,0,0.42
```
Upload the CSV from the sidebar. If `y_probability` is present, the threshold slider recalculates predicted labels.

### Important about the bundled data
`data/sample_predictions.csv` is synthetic demonstration data so the project runs immediately. It is not a result from your earlier trained model and must not be presented as actual model performance.

## Optional baseline training script
If you have the prepared churn dataset from the earlier project, save it as:
`data/customer_churn_analysis_ready.csv`

It must contain a binary `churn_flag` target (0 = active/no churn, 1 = churn). Run:
```bash
python src/train_model.py
```
The script creates `outputs/model_predictions.csv`. Upload that file in the dashboard. The script is a baseline example; to evaluate your exact earlier model, export predictions from that model instead.

## GitHub upload
1. Extract the ZIP file.
2. Open GitHub and create a new public repository, e.g. `customer-churn-plotly-dashboard`.
3. Open the repository and choose **Add file → Upload files**.
4. Upload the contents inside the extracted project folder (not the ZIP itself).
5. Commit changes with message: `Add interactive Plotly model evaluation dashboard`.

## Suggested GitHub description
Interactive customer churn model evaluation dashboard built with Streamlit and Plotly. Displays accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and classification report, with CSV upload support.
