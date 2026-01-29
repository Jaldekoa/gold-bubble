
# 🎈 Gold Bubble Analyzer 🎈

A Streamlit application that detects and visualizes speculative bubbles in gold futures (GC=F) using Log-Periodic Power Law Singularity (LPPLS) modeling.

## Features

- **Bubble Detection**: Identifies bubble formation start dates using LPPLS Lagrange-based analysis
- **Parameter Fitting**: Calculates optimal LPPLS parameters via CMA-ES optimization
- **Dual Visualization**: View price data in logarithmic and nominal USD scales
- **Key Metrics**: Displays bubble start date, predicted bursting date, and price at burst
- **Real-time Data**: Fetches live gold futures data from Yahoo Finance

## Installation

```bash
pip install streamlit lppls yfinance numpy pandas plotly
```

## Usage

```bash
streamlit run main.py
```

## How It Works

1. **Data Acquisition**: Fetches GC=F historical data starting from 2022-01-01
2. **Bubble Detection**: Uses Lagrange regularisation to find optimal bubble window and start time
3. **LPPLS Fitting**: Applies CMA-ES to fit the model parameters (tc, m, w, a, b, c1, c2)
4. **Visualization**: Plots actual vs. predicted prices with predicted crash date

## Key Parameters

- `tc`: Critical time (bursting date)
- `m`: Power law exponent
- `w`: Log-frequency of oscillations
- `a, b`: Amplitude parameters
- `c1, c2`: Phase parameters

## Caching

Results are cached for 24 hours to reduce API calls and computation time.
