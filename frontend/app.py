import streamlit as st
import requests
import os

# --- App title ---
st.title("🎬 Box Office Predictor")

# --- Backend URL (read from environment, fallback to localhost) ---
backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# --- User inputs ---
genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", "Others"])
budget = st.number_input("Budget (USD)", min_value=0.0, step=100000.0)
one_week_sales = st.number_input("First Week Sales (USD)", min_value=0.0, step=100000.0)
imdb_rating = st.number_input("IMDb Rating", min_value=0.0, max_value=10.0, step=0.1)
director = st.text_input("Director")
lead_actor = st.text_input("Lead Actor")

# --- Predict button ---
if st.button("Predict Box Office"):
    # Build payload for backend
    payload = {
        "Genre": genre,
        "BudgetUSD": budget,
        "One_Week_SalesUSD": one_week_sales,
        "IMDbRating": imdb_rating,
        "Director": director,
        "LeadActor": lead_actor
    }

    try:
        # Call backend API
        response = requests.post(f"{backend_url}/predict", json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract prediction safely
        prediction = data.get("prediction")
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            prediction = prediction[0]  # get the first element if returned as list
        if prediction is None:
            st.error("Backend returned no prediction.")
        else:
            st.success(f"💰 Predicted Box Office: ${prediction:,.2f}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error in prediction: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

