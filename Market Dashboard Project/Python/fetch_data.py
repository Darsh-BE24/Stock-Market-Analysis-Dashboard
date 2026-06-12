import yfinance as yf
import pandas as pd
import numpy as np

# Stock tickers
stocks = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "NIFTY50": "^NSEI"
}

all_data = []

# Download stock data
for name, ticker in stocks.items():

    print(f"Downloading {name}...")

    try:
        df = yf.download(
            ticker,
            period="1y",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            print(f"No data found for {name}")
            continue

        # Flatten columns immediately
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Reset index
        df.reset_index(inplace=True)

        # Add stock name
        df["Stock"] = name

        # Keep only required columns
        df = df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Stock"
            ]
        ]

        all_data.append(df)

        print(f"{name} downloaded successfully!")

    except Exception as e:
        print(f"Error downloading {name}: {e}")

# Check if data exists
if len(all_data) == 0:

    print("No data downloaded.")

else:

    # Combine all stock data
    final_df = pd.concat(all_data, ignore_index=True)

    # Sort data
    final_df = final_df.sort_values(
        by=["Stock", "Date"]
    )

    # =========================
    # DAILY RETURN %
    # =========================
    final_df["Daily Return %"] = (
        (final_df["Close"] - final_df["Open"])
        / final_df["Open"]
    ) * 100

    # =========================
    # VOLATILITY %
    # =========================
    final_df["Volatility %"] = (
        (final_df["High"] - final_df["Low"])
        / final_df["Open"]
    ) * 100

    # =========================
    # MA7
    # =========================
    final_df["MA7"] = (
        final_df.groupby("Stock")["Close"]
        .transform(lambda x: x.rolling(7).mean())
    )

    # =========================
    # MA30
    # =========================
    final_df["MA30"] = (
        final_df.groupby("Stock")["Close"]
        .transform(lambda x: x.rolling(30).mean())
    )

    # =========================
    # EMA20
    # =========================
    final_df["EMA20"] = (
        final_df.groupby("Stock")["Close"]
        .transform(lambda x: x.ewm(span=20).mean())
    )

    # =========================
    # ROLLING VOLATILITY
    # =========================
    final_df["Rolling Volatility"] = (
        final_df.groupby("Stock")["Daily Return %"]
        .transform(lambda x: x.rolling(14).std())
    )

    # =========================
    # SHARPE RATIO
    # =========================
    final_df["Sharpe Ratio"] = (
        final_df.groupby("Stock")["Daily Return %"]
        .transform(
            lambda x:
            x.rolling(14).mean() /
            x.rolling(14).std()
        )
    )

    # =========================
    # RSI FUNCTION
    # =========================
    def calculate_rsi(series, period=14):

        delta = series.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()

        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    # RSI
    final_df["RSI"] = (
        final_df.groupby("Stock")["Close"]
        .transform(lambda x: calculate_rsi(x))
    )

    # Export to Excel
    final_df.to_excel(
        "../data/market_data.xlsx",
        index=False
    )

    print("Excel file created successfully!")