# California House Price Prediction

A machine learning web application that predicts California housing values using a **Random Forest Regressor**, with a **FastAPI backend** and a **Streamlit frontend**.

## Features

- Predict a house value from eight California Housing dataset features.
- Upload a CSV file to generate predictions for multiple rows.
- Download the prediction results as a CSV file.
- FastAPI endpoints for predictions and service health.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Streamlit
- Requests

## Project Structure

```text
House_pred_api/
├── main.py                  # FastAPI backend
├── app.py                   # Streamlit frontend
├── house_model.joblib       # Trained Random Forest model
├── house_features.joblib    # Model feature names
├── test_house.csv           # Example CSV input (optional)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd House_pred_api
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands below using the Python interpreter from your environment, or follow your organization's policy for script execution.

### 3. Install dependencies

```powershell
python -m pip install fastapi uvicorn streamlit requests pandas scikit-learn joblib python-multipart
```

Alternatively, if a `requirements.txt` file is present:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Open **two VS Code terminals** in the project directory. Activate the virtual environment in both.

### Terminal 1 — Start the FastAPI backend

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

API root: `http://127.0.0.1:8001/`  
Interactive API documentation: `http://127.0.0.1:8001/docs`

### Terminal 2 — Start the Streamlit frontend

Make sure `API_URL` in `app.py` points to the same backend port:

```python
API_URL = "http://127.0.0.1:8001"
```

Then run:

```powershell
python -m streamlit run app.py
```

Streamlit normally opens at `http://localhost:8501`.

Keep both terminal processes running while using the app.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API status |
| `GET` | `/health` | Health and model information |
| `POST` | `/predict` | Predict one house value from JSON input |
| `POST` | `/predict_file` | Accept a CSV and return a CSV with predictions |

## Required CSV Columns

For `/predict_file`, the CSV must contain these columns:

```text
MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
```

The column names must match exactly. Extra columns are retained in the output by the current backend.

## Notes

- The model is trained using the Scikit-learn California Housing dataset.
- In this dataset, the target `MedHouseVal`/`Price` is expressed in units of **$100,000**. If the model was trained on the unmodified dataset target, convert predictions to dollars by multiplying by `100000`, not `1000000`. Keep the API's single-row and CSV conversion logic consistent.
- The error range shown by the API is currently a fixed estimate, not a calibrated confidence interval. Update it using measured evaluation results before presenting it as a confidence interval.
- Keep the API URL in `app.py` consistent with the port used to start Uvicorn.

## Troubleshooting

- **Connection refused:** Start the FastAPI backend first and keep its terminal open.
- **WinError 10013 on port 8000:** Try a permitted port such as `8001`; update `API_URL` in `app.py` to match.
- **Missing model files:** Run the app from the project directory and ensure `house_model.joblib` and `house_features.joblib` are available.
- **CSV validation error:** Check that all eight required column names are present.

## Future Improvements

- Add visualizations and prediction history.
- Improve input validation and user experience.
- Add automated tests and deployment configuration.
