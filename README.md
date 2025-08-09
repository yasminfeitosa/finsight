# 📊 FinDocGPT – AI for Financial Insights & Strategy

FinDocGPT is a hackathon project that:
- Reads and analyzes financial reports.
- Extracts key metrics (revenue, profit, etc.).
- Performs sentiment analysis.
- Forecasts stock prices.
- Recommends Buy / Sell / Hold actions.

Built with:
- **Python** + **Streamlit** for the app.
- **OpenAI API** for document Q&A.
- **TextBlob** for sentiment.
- **Prophet** + **yfinance** for forecasting.

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yasminfeitosa/finsight
cd finsight
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set your OpenAI API Key
.env:
```bash
OPENAI_API_KEY=your_api_key_here
```

Or export it in terminal:

```bash
export OPENAI_API_KEY="your_api_key_here"   # Mac/Linux
setx OPENAI_API_KEY "your_api_key_here"     # Windows
```

### 5️⃣ Run the app
```bash
streamlit run app.py
```

## Project Structure
```bash
fin_doc_gpt/
│
├── app.py                # Streamlit UI
├── nlp_module.py         # Stage 1: Q&A, sentiment, anomalies
├── forecast_module.py    # Stage 2: stock price forecasting
├── strategy_module.py    # Stage 3: investment recommendation
├── requirements.txt
├── README.md
└── .env                  # Your API key (not committed to Git)
```

## 📌 Usage
- Upload an earnings report (.txt file).
- Enter a stock ticker (e.g., AAPL).
View:
- Key financial metrics from the document.
- Sentiment score.
- Price forecast chart.
- Investment recommendation (BUY/SELL/HOLD).