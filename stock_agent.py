import logging
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, render_template, request

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Global model variable
model = None

def get_stock_data(stock_symbol):
    """Fetch historical stock data using Yahoo Finance."""
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="6mo")  # Fetch last 6 months of data
        
        # Log if no data was found
        if data.empty:
            raise ValueError(f"No data found for {stock_symbol}")
        
        # Log the raw data for better visibility
        logging.debug(f"Raw data fetched for {stock_symbol}: {data.head()}")
        
        # Processing data
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Log the processed data
        logging.debug(f"Processed data for {stock_symbol}: {data.head()}")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None


def prepare_data(stock_symbol):
    """Prepare data for model training and prediction."""
    stock_data = get_stock_data(stock_symbol)
    
    if stock_data is None:
        logging.error(f"Failed to get stock data for {stock_symbol}. Returning None.")
        return None, None, None, None

    # Feature selection
    stock_data['Day'] = stock_data['Date'].dt.day
    stock_data['Prev Close'] = stock_data['Close'].shift(1)
    stock_data['Moving Average'] = stock_data['Close'].rolling(window=5).mean()
    
    # Log the features after engineering
    logging.debug(f"Features after engineering: {stock_data[['Day', 'Prev Close', 'Moving Average']].head()}")
    
    stock_data.dropna(inplace=True)  # Remove any rows with NaN values
    
    logging.debug(f"Cleaned features after dropping NaN: {stock_data[['Day', 'Prev Close', 'Moving Average']].head()}")
    
    X = stock_data[['Day', 'Prev Close', 'Moving Average']]
    y = stock_data['Close']

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, y_train, scaler, stock_data


def load_or_train_model(X_train, y_train, scaler):
    """Load an existing model or train a new one."""
    global model
    if model is None:
        logging.info("Training new model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save the model for future use
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
            pickle.dump(scaler, f)
    else:
        logging.info("Loading existing model...")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            scaler = pickle.load(f)
    
    return model


def predict_stock_prices(stock_symbol, num_days=365):
    """Use the model to predict stock prices for the next specified number of days."""
    try:
        X_train, y_train, scaler, stock_data = prepare_data(stock_symbol)

        if X_train is None:
            logging.error(f"Failed to prepare data for prediction for {stock_symbol}.")
            return None, None

        model = load_or_train_model(X_train, y_train, scaler)

        # Generate future dates for `num_days`
        future_days = np.array(range(1, num_days + 1)).reshape(-1, 1)

        # Simulate the 'Prev Close' and 'Moving Average' for future days
        last_row = stock_data.iloc[-1]
        prev_close = last_row['Close']
        moving_avg = last_row['Moving Average']
        
        # Simulate the missing features for future days
        future_data = np.array([[day, prev_close, moving_avg] for day in range(1, num_days + 1)])
        
        # Scale the future data with the same scaler used during training
        future_days_scaled = scaler.transform(future_data)

        # Predict
        predictions = model.predict(future_days_scaled)
        
        # Log the predictions to verify
        logging.debug(f"Predictions for {stock_symbol}: {predictions[:5]}")  # Log first 5 predictions

        # Create future date labels
        last_date = stock_data['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

        return future_dates, predictions

    except Exception as e:
        logging.error(f"Error in predicting stock prices for {stock_symbol}: {e}")
        return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        future_dates, predicted_prices = predict_stock_prices(stock_symbol)

        if future_dates is None or predicted_prices is None:
            return f"Could not retrieve data or predict for stock symbol: {stock_symbol}"
        
        # Prepare data for rendering in the template
        predictions_data = list(zip(future_dates, predicted_prices))
        return render_template("index.html", predictions=predictions_data)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
