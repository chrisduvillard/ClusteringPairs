# 📈 Futures Market Pair Detector

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges if applicable, e.g., build status -->

## Overview

This project presents an interactive web application built with Streamlit for identifying potential trading pairs within the futures market. It leverages high-quality historical data from Norgate Data and employs a suite of statistical and machine learning techniques to uncover relationships between different futures contracts.

The primary goal is to provide traders and analysts with a powerful tool to:
*   Discover statistically significant pairs for potential pairs trading strategies.
*   Explore correlations and co-movements between assets.
*   Gain insights into market dynamics and hedging opportunities.

This application demonstrates proficiency in data analysis, application development with Streamlit, integration with external data providers, and implementation of various quantitative techniques.

## Key Features

*   **Multiple Analysis Techniques:** Implements several distinct methods for pair discovery:
    *   🪙 **Cointegration (Engle-Granger):** Tests for long-run equilibrium.
    *   🔗 **Correlation (Returns):** Measures linear relationships between returns.
    *   📏 **Distance (SSD):** Finds pairs with similar normalized price shapes.
    *   🧩 **Clustering (K-Means + SSD Rank):** Groups instruments by price patterns and ranks pairs within clusters.
    *   〰️ **Similarity (DTW):** Measures shape similarity allowing for time shifts.
    *   ℹ️ **Mutual Information (Prices):** Detects linear and non-linear dependence.
*   **Norgate Data Integration:** Seamlessly fetches and processes continuous futures contract data. *(Requires a Norgate Data subscription)*.
*   **Interactive Streamlit UI:**
    *   Intuitive controls for date range selection, instrument choice, and technique configuration.
    *   Adjustable parameters for fine-tuning analysis (e.g., p-value, correlation threshold).
*   **Advanced Filtering:** Refine results based on data overlap and spread characteristics.
*   **Clear Visualizations:** Interactive Plotly charts for normalized prices and pair spreads.
*   **Technique Comparison:** Side-by-side comparison of results from different methods.
*   **In-App Documentation:** Explanations, pros/cons, and use cases for each technique.

## Technologies Used

*   **Backend:** Python
*   **Web Framework:** Streamlit
*   **Data Analysis:** Pandas, NumPy, Statsmodels, Scikit-learn
*   **Visualization:** Plotly
*   **Time Series Analysis:** dtaidistance (for Dynamic Time Warping)
*   **Data Provider:** NorgateData API

## Screenshots

**Main Interface**
![Main Interface](docs/images/main_interface.png)

**Pair Results Table**
![Table of Identified Pairs](docs/images/pair_table.png)

**Pair Visualization**
![Price Chart & Spread Chart](docs/images/pair_visualization.png)

**Technique Comparison Tab**
![Comparison of Multiple Techniques](docs/images/comparison.png)

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+ (as specified in `.python-version`)
    *   **Norgate Data Subscription:** Access to Norgate Data is required. Ensure the `norgatedata` library is installed and configured according to their instructions.
    *   Git (for cloning)

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL
    cd ClusteringPairs
    ```

3.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `norgatedata` is installed and licensed separately)*

5.  **Configure `app.py`:**
    *   Open `app.py`.
    *   **Crucial:** Populate the `TICKER_MAP` dictionary with your desired Norgate tickers (e.g., `&ES_CCB`) and display names (e.g., `E-mini S&P 500`).
    *   Set the `DEFAULT_WATCHLIST` variable to your Norgate watchlist name.

## Usage

1.  Ensure Norgate Data is configured and accessible.
2.  Activate your virtual environment.
3.  Run the application:
    ```bash
    streamlit run app.py
    ```
4.  Navigate the application using the sidebar to select instruments, dates, and analysis techniques. Explore the results in the main tabs.

## Potential Future Improvements

*   Integration with a backtesting framework (e.g., `backtrader`, `zipline-reloaded`) to evaluate strategy performance.
*   Addition of more advanced statistical tests or machine learning models (e.g., Kalman Filters, VECM).
*   Real-time data processing capabilities.
*   Deployment to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS).
*   User account system for saving configurations or results.

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (assuming you add an MIT license file).