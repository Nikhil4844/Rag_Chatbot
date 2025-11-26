import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt

def transaction_sentence(tr):
    return f"On {tr['date']}, {tr['customer']} purchased a {tr['product']} for ₹{tr['amount']}."

def process_json(json_obj):
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)
    assert isinstance(json_obj, list), "JSON must be a list of transaction dicts"
    return json_obj

st.title("RAG Chatbot - Upload any transaction JSON")

uploaded_file = st.file_uploader("Upload a JSON file containing transactions", type=["json"])
data = None

if uploaded_file:
    try:
        json_obj = json.load(uploaded_file)
        data = process_json(json_obj)
        st.success("JSON file loaded and parsed successfully.")
    except Exception as e:
        st.error(f"Error loading JSON: {e}")

if data:
    texts = [transaction_sentence(tr) for tr in data]
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()

    query = st.text_input("Ask your question about the transactions")
    top_k = st.slider("Number of retrieved results", 1, 5, 3)

    answer, show_info = None, False
    if query:
        query_emb = vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_emb, embeddings)[0]
        results = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)[:top_k]
        customers = list({tr['customer'] for tr in data})
        months = sorted({tr['date'][:7] for tr in data})

        q = query.lower()
        if max(sims) < 0.25:
            answer = None
            show_info = True
        else:
            if "total spending" in q:
                for cust in customers:
                    if cust.lower() in q:
                        total = sum(tr['amount'] for tr in data if tr['customer'].lower() == cust.lower())
                        answer = f"{cust} spent ₹{total}."
            elif "purchase history" in q:
                for cust in customers:
                    if cust.lower() in q:
                        purchases = [f"{tr['product']} for ₹{tr['amount']} on {tr['date']}" for tr in data if tr['customer'].lower() == cust.lower()]
                        answer = f"{cust}'s purchases: " + ", ".join(purchases)
            elif "transactions in" in q or "monthly transaction" in q:
                for month in months:
                    if month in q:
                        trans = [transaction_sentence(tr) for tr in data if tr['date'].startswith(month)]
                        answer = "Transactions in " + month + ": " + "; ".join(trans)
            elif "average transaction" in q:
                amounts = [tr['amount'] for tr in data]
                avg = round(sum(amounts)/len(amounts),2)
                answer = f"Average transaction amount is ₹{avg}."
            elif "most often" in q or "most common product" in q or "most frequently purchased product" in q:
                prod = Counter(tr['product'] for tr in data).most_common(1)[0][0]
                answer = f"Product purchased most often is {prod}."

        if answer:
            st.success(answer)
        elif show_info:
            st.info("No relevant information found for your query in the uploaded data.")

    st.markdown("## Charts (auto-update based on uploaded file)")
    # Spend per month
    months_chart = [tr['date'][:7] for tr in data]
    month_spend = {}
    for m in set(months_chart):
        month_spend[m] = sum(tr['amount'] for tr in data if tr['date'].startswith(m))
    fig, ax = plt.subplots()
    ax.bar(month_spend.keys(), month_spend.values())
    ax.set_title("Spend per Month")
    st.pyplot(fig)

    # Most frequent product
    prod_counts = Counter(tr['product'] for tr in data)
    fig2, ax2 = plt.subplots()
    ax2.pie(prod_counts.values(), labels=prod_counts.keys(), autopct="%1.1f%%")
    ax2.set_title("Most Frequently Purchased Product")
    st.pyplot(fig2)

else:
    st.info("Upload a JSON file to get started.")

