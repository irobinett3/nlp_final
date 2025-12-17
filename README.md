# RiskRadar 🔍

Automatically detect financial risks in corporate documents using NLP. Built for my Natural Language Processing final project at Notre Dame.

## What is this?

Companies publish tons of documents - SEC filings, earnings calls, press releases, news articles. Buried in all that text are warnings about financial trouble: lawsuits, debt problems, regulatory issues, you name it.

RiskRadar reads through these documents and flags the risky ones. It uses three different AI models that vote on how risky each document is, then trains a custom model to get really good at spotting trouble.

**The result?** 98.3% accuracy at detecting financial risks. The system can process 10,000 documents in about 15 minutes.

## How it works

### Part 1: Get the data (`data_pipeline/`)

Scrapes and stores documents from multiple sources:
- **SEC filings** (10-Ks, 10-Qs) - the official financial reports
- **Earnings call transcripts** - what executives say to investors
- **News articles** - using MarketAux API
- **Press releases** - from company websites

Everything gets stored in a SQLite database for easy access.

### Part 2: Label and train (`training_pipeline/`)

The magic happens here:

1. **Multi-source labeling** - Three different models analyze each document:
   - FinBERT (financial sentiment model)
   - Financial PhraseBank (trained on financial news)
   - GPT-4o-mini (as a sanity check)

2. **Consensus voting** - If 2+ models agree, we trust that label. If they disagree, we flag it for review.

3. **BIO tagging** - We don't just say "this is risky" - we identify the exact sentences and phrases that signal risk.

4. **Model training** - Take all that labeled data and fine-tune a transformer model to be really good at this specific task.

## Quick Start

### You'll need:
- Python 3.8+
- An OpenAI API key (for the LLM labeling)
- Your email for SEC (they require it)

### Setup (5 minutes):

```bash
# 1. Install dependencies
pip install -r training_pipeline/requirements.txt
pip install -r data_pipeline/requirements.txt

# Or install everything:
pip install requests beautifulsoup4 lxml python-dotenv aiohttp torch transformers scikit-learn pandas numpy tqdm openai anthropic

# 2. Create a .env file with your keys
echo 'SEC_USER_AGENT="YourName/1.0 (your.email@example.com)"' > .env
echo 'OPENAI_API_KEY="sk-your-key-here"' >> .env

# 3. Optional: Get a free MarketAux API token for news
# Add to .env: MARKETAUX_API_TOKEN="your-token"
```

### Get some data:

```bash
cd data_pipeline

# Grab documents for one company (quick test)
python ingest_data_bulk.py --company "Apple Inc." --ticker AAPL --years 2

# Or go big - top 50 S&P 500 companies
python ingest_data_bulk.py --top-sp500 50 --max
```

This creates a SQLite database called `riskradar_ingest.db` with all the documents.

### Train a model:

```bash
cd training_pipeline

# Fast mode: Process 5000 docs in ~10 minutes (no LLM calls = free)
python -m training_pipeline.fast_pipeline --limit 5000

# Full mode: Includes LLM validation and BIO tagging (~$2-3 for 5000 docs)
python -m training_pipeline.pipeline --limit 5000

# This creates training_data/run_YYYYMMDD_HHMMSS/ with your labeled datasets
```

### Train your classifier:

```bash
python -m training_pipeline.train_model \
    --train-data training_data/run_*/classification_train.json \
    --val-data training_data/run_*/classification_val.json \
    --output-dir models/my_model
```

Takes about 10-20 minutes depending on dataset size and whether you have a GPU.

## Project Structure

```
riskradar/
├── data_pipeline/              # Scraping and data storage
│   ├── sec_ingestion.py        # Get SEC filings
│   ├── transcript_ingestion.py # Get earnings calls
│   ├── news_ingestion.py       # Get news articles
│   ├── press_release_ingestion.py
│   ├── ingest_data_bulk.py     # Main entry point
│   └── database.py             # SQLite operations
│
└── training_pipeline/          # Labeling and model training
    ├── labelers.py             # Multi-source labeling logic
    ├── fast_pipeline.py        # Quick labeling (no LLM)
    ├── pipeline.py             # Full pipeline (with LLM + BIO tags)
    ├── train_model.py          # Train the classifier
    ├── train_span_detector.py  # Train span detection model
    └── compare_models.py       # Evaluate results
```

## Risk Categories

The system classifies documents into four buckets:

- **high_risk** - Bankruptcy, fraud, major lawsuits, severe financial distress
- **medium_risk** - Revenue decline, regulatory issues, operational problems  
- **low_risk** - Minor concerns, competitive pressures, market uncertainties
- **no_risk** - Neutral or positive content

## Results

On a test set of ~1000 documents:
- **Accuracy**: 98.3%
- **Precision**: 97.8%
- **Recall**: 98.1%

The span detection model (which highlights risky phrases) gets:
- **Token accuracy**: 95.2%
- **Exact span match**: 76.4%

## Cost & Performance

**Data ingestion**: Free (just time and bandwidth)
- Single company: ~2-5 minutes
- Top 50 S&P 500: ~1-2 hours

**Fast pipeline** (no LLM): Free
- 10,000 documents: ~15 minutes on 8-core CPU

**Full pipeline** (with GPT-4o-mini): ~$0.50 per 1000 documents
- 5,000 documents: ~3.5 hours, ~$2.50

**Training**: Free (unless you count electricity)
- ~10-20 minutes with GPU
- ~1-2 hours with CPU

## Common Issues

**"403 Forbidden" from SEC**
- Add your real email to `SEC_USER_AGENT` in `.env`
- SEC requires this - format: `"YourName/1.0 (you@example.com)"`

**Out of memory**
- Reduce the `--limit` parameter
- Or use `fast_pipeline.py` instead of full `pipeline.py`

**Slow on first run**
- First time downloads ~2GB of AI models (FinBERT, etc.)
- They get cached, so subsequent runs are fast

**OpenAI API errors**
- Make sure your API key is valid and has credits
- Check https://platform.openai.com/usage

## Files You Shouldn't Commit to Git

Create a `.gitignore` with at least:
```
.env
*.db
__pycache__/
training_data/
models/
venv/
```

Seriously - don't commit your `.env` file or you'll leak your API keys!

## What's Next?

Some ideas if you want to extend this:
- Add more data sources (Twitter/X, Reddit, analyst reports)
- Try different model architectures (RoBERTa, DeBERTa)
- Build a web interface for exploring results
- Create real-time monitoring for specific companies
- Compare risk detection across industries

## Credits

**Author**: Ian Robinet (irobinet@nd.edu)  
**Course**: CSE 40657 Natural Language Processing, Notre Dame  
**Models Used**:
- FinBERT: yiyanghkust/finbert-tone
- Financial PhraseBank: ProsusAI/finbert
- GPT-4o-mini via OpenAI API