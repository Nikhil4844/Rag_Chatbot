# Minimal RAG Chatbot for Transactional Data (Low RAM Render Deployment)

**Purpose:**  
Python chatbot answering business queries about customer transactions using TF-IDF embeddings (no heavy ML models). Deployable on Render free tier (512MB RAM limit).

## Usage

1. Clone repo, ensure `transactions.json` and all `.py` files are present.
2. Install requirements:
pip install -r requirements.txt
3. Run CLI:  
python rag_chatbot.py
4. Run Streamlit:  
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
5. **No-cost Render deployment:** Just add your repo, set above start command.

## Example queries

- What is Amit's total spending?  
- Show Riya’s purchase history.  
- Average transaction amount?  
- Transactions in 2024-02?  
- Which product was purchased most often?

## Features

- TF-IDF embeddings for minimum RAM.
- Streamlit UI for web.
- Charts: monthly spend, product frequency.
- No background workers, small dependencies.
