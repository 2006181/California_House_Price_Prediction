import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# FastAPI backend URL
API_URL = "http://127.0.0.1:8001"

# Page title
st.title("🏡 California House Price Prediction")
st.write(
    "Predict California house prices using Machine Learning "
    "and Random Forest Regression."
)

st.divider()


# Check backend status
try:
    response = requests.get(f"{API_URL}/health", timeout=5)

    if response.status_code == 200:
        st.success("Backend Connected Successfully ✅")
    else:
        st.warning("Backend is running but returned an unexpected response.")

except requests.exceptions.RequestException:
    st.error(
        "Backend is not connected. "
        "Please start your FastAPI server first."
    )


# Create tabs
tab1, tab2 = st.tabs([
    "🏠 Single House Prediction",
    "📂 CSV Bulk Prediction"
])


# ==========================================
# TAB 1: SINGLE HOUSE PREDICTION
# ==========================================

with tab1:

    st.subheader("Enter House Details")

    with st.form("house_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            MedInc = st.number_input(
                "Median Income (in $10,000 units)",
                min_value=0.01,
                value=5.0,
                step=0.1
            )

            HouseAge = st.number_input(
                "House Age (years)",
                min_value=0.01,
                value=25.0,
                step=1.0
            )

            AveRooms = st.number_input(
                "Average Rooms",
                min_value=0.01,
                value=5.0,
                step=0.1
            )

            AveBedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.01,
                value=1.0,
                step=0.1
            )

        with col2:
            Population = st.number_input(
                "Population",
                min_value=0.01,
                value=1000.0,
                step=10.0
            )

            AveOccup = st.number_input(
                "Average Occupancy",
                min_value=0.01,
                value=3.0,
                step=0.1
            )

            Latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=36.0,
                step=0.1
            )

            Longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-120.0,
                step=0.1
            )

        predict_button = st.form_submit_button(
            "🔍 Predict House Price",
            use_container_width=True
        )

    if predict_button:

        house_data = {
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }

        try:
            with st.spinner("Predicting house price..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=house_data,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction Completed Successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted House Price",
                        result["predicted_price"]
                    )

                with col2:
                    st.metric(
                        "Estimated Error Range",
                        "$39,000"
                    )

                st.info(
                    f"Reported range: "
                    f"{result['confidence_range']}"
                )

                st.caption(
                    "The reported range is based on the backend's "
                    "fixed $39,000 estimate; it is not a calibrated "
                    "confidence interval."
                )

            else:
                st.error(
                    f"Prediction failed: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to FastAPI: {e}")


# ==========================================
# TAB 2: CSV BULK PREDICTION
# ==========================================

with tab2:

    st.subheader("Upload CSV File")

    st.write(
        "Upload a CSV containing the eight required house features. "
        "The API will predict prices for all rows."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.info(f"Selected file: {uploaded_file.name}")

        if st.button("📊 Predict All House Prices"):

            try:
                with st.spinner("Processing CSV file..."):

                    response = requests.post(
                        f"{API_URL}/predict_file",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "text/csv"
                            )
                        },
                        timeout=120
                    )

                if response.status_code == 200:

                    st.success(
                        "Predictions generated successfully! ✅"
                    )

                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=response.content,
                        file_name="prediction.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:
                    st.error(
                        f"CSV prediction failed: {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to FastAPI: {e}")


# Footer
st.divider()

st.caption(
    "Built with Python, Streamlit, FastAPI and "
    "Random Forest Regression."
)