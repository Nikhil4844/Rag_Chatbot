import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt

def transaction_sentence(tr):
    return f"On {tr['date']}, {tr['customer']} purchased a {tr['product']} for ₹{tr['amount']}."

def process_json(json_obj):
    # Check structure
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)
    assert isinstance(json_obj, list), "JSON must be a list of transaction dicts"
    return json_obj

st.title("Minimal RAG Chatbot (Upload Your Own Transaction JSON!)")

uploaded_file = st.file_uploader("Upload a JSON file containing transactions", type=["json"])
use_default = False
data = None

if uploaded_file:
    try:
        json_obj = json.load(uploaded_file)
        data = process_json(json_obj)
        st.success("JSON file loaded and parsed successfully.")
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
else:
    if st.button("Use Example Transactions"):
        use_default = True
        data = [
            {"id": 1, "customer": "Amit", "product": "Laptop", "amount": 55000, "date": "2024-01-12"},
            {"id": 2, "customer": "Amit", "product": "Mouse", "amount": 700, "date": "2024-02-15"},
            {"id": 3, "customer": "Riya", "product": "Mobile", "amount": 30000, "date": "2024-01-05"},
            {"id": 4, "customer": "Riya", "product": "Earbuds", "amount": 1500, "date": "2024-02-20"},
            {"id": 5, "customer": "Karan", "product": "Keyboard", "amount": 1200, "date": "2024-03-01"}
        ]
        st.info("Default dataset loaded for demo.")

if data:
    texts = [transaction_sentence(tr) for tr in data]
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()

    st.subheader("Upload-based Transactions")
    for t in texts:
        st.write(t)

    query = st.text_input("Ask your question about these transactions")
    top_k = st.slider("Number of retrieved results", 1, 5, 3)

    if query:
        query_emb = vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_emb, embeddings)[0]
        results = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)[:top_k]
        st.markdown("**Snippets & scores:**")
        for txt, score in results:
            st.write(f'"{txt}" [score: {score:.2f}]')
        if max(sims) < 0.25:
            st.warning("I don't have enough information in the uploaded documents.")
        else:
            # Deterministic answer logic for generic data
            q = query.lower()
            ans = None
            customers = list({tr['customer'] for tr in data})
            months = sorted({tr['date'][:7] for tr in data})
            if "total spending" in q:
                for cust in customers:
                    if cust.lower() in q:
                        total = sum(tr['amount'] for tr in data if tr['customer'].lower() == cust.lower())
                        ans = f"{cust} spent ₹{total}."
            elif "purchase history" in q:
                for cust in customers:
                    if cust.lower() in q:
                        purchases = [f"{tr['product']} for ₹{tr['amount']} on {tr['date']}" for tr in data if tr['customer'].lower() == cust.lower()]
                        ans = f"{cust}'s purchases: " + ", ".join(purchases)
            elif "transactions in" in q or "monthly transaction" in q:
                for month in months:
                    if month in q:
                        trans = [transaction_sentence(tr) for tr in data if tr['date'].startswith(month)]
                        ans = "Transactions in " + month + ": " + "; ".join(trans)
            elif "average transaction" in q:
                amounts = [tr['amount'] for tr in data]
                avg = round(sum(amounts)/len(amounts),2)
                ans = f"Average transaction amount is ₹{avg}."
            elif "most often" in q or "most common product" in q or "most frequently purchased product" in q:
                prod = Counter(tr['product'] for tr in data).most_common(1)[0][0]
                ans = f"Product purchased most often is {prod}."
            if ans:
                st.success(ans)
            else:
                st.warning("I don't have enough information in the uploaded documents.")

    st.markdown("## Charts")
    # Spend per month
    months = [tr['date'][:7] for tr in data]
    month_spend = {}
    for m in months:
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
    st.info("Upload a JSON file to get started, or click 'Use Example Transactions' for demo.")

