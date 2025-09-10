# Stock Price Prediction using LSTM (PyTorch)

This project uses a Long Short-Term Memory (LSTM) neural network implemented in PyTorch to predict stock prices based on historical closing price data.

---

## Project Structure

- `share_price_prediction.py`: The main Python script containing the data preprocessing, LSTM model, training, and evaluation code.
- `symbols_valid_meta.csv`: Your CSV file containing stock data with a **Close** price column.

---

## Prerequisites

Make sure you have Python 3.7+ installed. It is recommended to use a virtual environment.

### Install required packages

```bash
pip install numpy pandas scikit-learn torch


How to Run

Place your CSV file (symbols_valid_meta.csv) in the folder:

/home/great/Documents/share price predection/2/symbols_valid_meta.csv


(or update the path in the script accordingly).

Run the script:

python3 share_price_prediction.py


The script will:

Print the columns in the CSV and the first 5 rows (to help you verify the data).

Train an LSTM model on 80% of the data.

Evaluate and print predicted vs actual prices on the remaining 20%.

Notes

The script assumes your CSV contains a column named 'Close' (case insensitive). If your CSV has a different name for closing prices, update the script accordingly.

This is a simple baseline model and does not guarantee accurate stock predictions. Stock prices are influenced by many external factors.

Feel free to modify the sequence length, model parameters, and training epochs as needed.
