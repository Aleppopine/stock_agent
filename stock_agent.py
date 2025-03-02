import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
import logging
import time

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "stock_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def get_stock_data(stock_symbol):
    """Fetch historical stock data using Yahoo Finance."""
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="6mo")  # Fetch last 6 months of data
        if data.empty:
            raise ValueError(f"No data found for {stock_symbol}")
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        logging.debug(f"Fetched data for {stock_symbol}: {data.head()}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def prepare_data(stock_symbol):
    """Prepare data for model training and prediction."""
    stock_data = get_stock_data(stock_symbol)
    
    if stock_data is None:
        return None, None, None, None

    # Feature selection
    stock_data['Day'] = stock_data['Date'].dt.day
    X = stock_data[['Day']]
    y = stock_data['Close']

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, y_train, scaler, stock_data

def load_or_train_model(X_train, y_train, scaler):
    """Load existing model or train a new one if not found."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        with open(SCALER_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        logging.info("Loaded existing model from disk.")
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Ensure model directory exists
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # Save model and scaler
        with open(MODEL_PATH, "wb") as model_file:
            pickle.dump(model, model_file)
        with open(SCALER_PATH, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        logging.info("Trained and saved a new model.")
    return model

def update_model(stock_symbol):
    """Update the existing model with new stock data."""
    X_train, y_train, scaler, stock_data = prepare_data(stock_symbol)
    if X_train is None:
        return None
    
    model = load_or_train_model(X_train, y_train, scaler)
    model.fit(X_train, y_train)  # Retrain model with new data
    
    # Save updated model
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info("Updated and saved model with new data.")
    return model

def predict_stock_prices(stock_symbol, num_days=365):
    """Use the model to predict stock prices for the next specified number of days."""
    X_train, y_train, scaler, stock_data = prepare_data(stock_symbol)

    if X_train is None:
        return None, None

    model = update_model(stock_symbol)

    # Generate future dates for `num_days`
    future_days = np.array(range(1, num_days + 1)).reshape(-1, 1)
    future_days_scaled = scaler.transform(future_days)

    # Predict
    predictions = model.predict(future_days_scaled)

    # Create future date labels
    last_date = stock_data['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

    return future_dates, predictions

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    stock_symbol = None
    error_message = None

    if request.method == "POST":
        # Check which form was submitted: prediction or update model
        if "stock_symbol" in request.form:
            # Handle prediction request
            stock_symbol = request.form["stock_symbol"].upper()
            future_dates, predicted_prices = predict_stock_prices(stock_symbol)

            if future_dates is None:
                error_message = "Stock symbol not found or no data available."
            else:
                predictions = list(zip(future_dates, predicted_prices))

                # Ensure 'static' directory exists
                if not os.path.exists("static"):
                    os.makedirs("static")

                # Plot predictions
                plt.figure(figsize=(10, 5))
                plt.plot(future_dates, predicted_prices, marker='o', linestyle='-', color='blue', label="Predicted Prices")
                plt.xlabel("Date")
                plt.ylabel("Stock Price")
                plt.legend()
                plt.grid()
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save plot with a unique filename using timestamp
                plot_filename = f"static/stock_plot_{int(time.time())}.png"
                plt.savefig(plot_filename)
                plt.close()
                plt.clf()

        elif "update_model" in request.form:
            # Handle update model request
            stock_symbol = request.form.get("stock_symbol", "AAPL").upper()  # Default to AAPL if none provided
            logging.info(f"Triggering model update for {stock_symbol}...")

            model = update_model(stock_symbol)  # Update the model

            if model is None:
                error_message = "Model update failed. Please try again later."
            else:
                logging.info(f"Model for {stock_symbol} updated successfully.")
                error_message = "Model updated successfully!"  # Show success message
    
    return render_template("index.html", stock_symbol=stock_symbol, predictions=predictions, error_message=error_message)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's dynamic port
    app.run(host="0.0.0.0", port=port)  # Don't use debug=True in production
