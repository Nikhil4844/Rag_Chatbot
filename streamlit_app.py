import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt

def load_transactions():
    with open('transactions.json','r') as f:
        return json.load(f)

def transaction_sentence(tr):
    return f"On {tr['date']}, {tr['customer']} purchased a {tr['product']} for ₹{tr['amount']}."

data = load_transactions()
texts = [transaction_sentence(tr) for tr in data]
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(texts).toarray()

st.title("Minimal Transaction RAG Chatbot (TF-IDF, Low RAM)")
st.write("Transactions:")
for t in texts:
    st.write(t)

query = st.text_input("Your question")
top_k = st.slider("Top K retrieved", 1, 5, 3)
if query:
    query_emb = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_emb, embeddings)[0]
    results = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)[:top_k]
    st.write("Snippets & scores:")
    for txt, score in results:
        st.write(f'"{txt}" [score: {score:.2f}]')
    if max(sims)<0.25:
        st.warning("I don't have enough information in the uploaded documents.")
    else:
        # Deterministic answer logic
        q = query.lower()
        ans = None
        if "total spending" in q:
            for cust in ['Amit', 'Riya', 'Karan']:
                if cust.lower() in q:
                    total = sum(tr['amount'] for tr in data if tr['customer'].lower()==cust.lower())
                    ans = f"{cust} spent ₹{total}."
        elif "purchase history" in q:
            for cust in ['Amit','Riya','Karan']:
                if cust.lower() in q:
                    purchases = [f"{tr['product']} for ₹{tr['amount']} on {tr['date']}" for tr in data if tr['customer'].lower()==cust.lower()]
                    ans = f"{cust}'s purchases: " + ", ".join(purchases)
        elif "transactions in" in q or "monthly transaction" in q:
            for month in ['2024-01','2024-02','2024-03']:
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
month_spend = {m:0 for m in set(months)}
for m in months:
    month_spend[m] += sum(tr['amount'] for tr in data if tr['date'].startswith(m))
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
