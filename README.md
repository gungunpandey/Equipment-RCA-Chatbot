# Equipment RCA Assistant

A conversational AI chatbot that guides engineers through structured Root Cause Analysis (RCA) for industrial equipment failures.

## What it does

- Collects fault information through a chat conversation
- Runs domain expert analysis (Mechanical / Electrical / Process)
- Performs 5 Whys root cause drill-down
- Supports image uploads for visual damage analysis
- Retrieves context from OEM equipment manuals via Weaviate RAG
- Falls back to web search if no manual is found

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in this folder:
   ```
   OPENROUTER_API_KEY=your_key
   OPENROUTER_MODEL=openai/gpt-5.2
   WEAVIATE_URL=your_url
   WEAVIATE_API_KEY=your_key
   WEAVIATE_COLLECTION=Rca
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

## Project Structure

```
AI_UseCase/
├── app.py              # Main Streamlit app
├── config/config.py    # API keys and settings
├── models/llm.py       # LLM setup (OpenRouter)
├── utils/
│   ├── rag_utils.py    # Weaviate RAG retrieval
│   ├── web_search.py   # DuckDuckGo web search
│   ├── rca_utils.py    # RCA prompts and parsing
│   ├── image_utils.py  # Vision model for image analysis
│   └── pdf_utils.py    # PDF text extraction
└── reference/          # PPT content and notes
```
