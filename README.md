# Bitcoin Price Prediction using LSTM

## Overview
This project uses an LSTM (Long Short-Term Memory) neural network to predict Bitcoin prices based on historical data from Yahoo Finance. The model is trained on past price movements and makes future predictions.

## Features
- Fetches historical Bitcoin price data from Yahoo Finance
- Preprocesses the data using MinMax scaling
- Creates time-series sequences for training
- Splits data sequentially to avoid data leakage
- Builds and trains an LSTM model with dropout layers to prevent overfitting
- Evaluates model performance using RMSE (Root Mean Squared Error)
- Plots actual vs. predicted Bitcoin prices

## Installation
Ensure you have Python up to v3.11 installed, then install dependencies using:
```bash
pip3 install -r requirements.txt
```

## Usage
Run the script to train the model and generate predictions:
```bash
python3 analyst.py
```

## Dependencies
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

## Model Performance
The model's performance is measured using RMSE, which is printed after training. A lower RMSE indicates better accuracy.

## Visualization
The script plots actual vs. predicted Bitcoin prices, allowing you to visually assess the model's accuracy.

## License
This project is licensed under the MIT License.

