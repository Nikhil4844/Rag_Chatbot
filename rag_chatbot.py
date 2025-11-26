import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_transactions(filename='transactions.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def transaction_sentence(tr):
    return f"On {tr['date']}, {tr['customer']} purchased a {tr['product']} for ₹{tr['amount']}."

def preprocess_transactions(data):
    return [transaction_sentence(tr) for tr in data]

def create_embeddings(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(texts).toarray()

def embed_query(query, vectorizer):
    return vectorizer.transform([query]).toarray()

def retrieve_transactions(query, embeddings, texts, top_k=3, vectorizer=None):
    q_emb = embed_query(query, vectorizer)
    sims = cosine_similarity(q_emb, embeddings)[0]
    sorted_pairs = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:top_k]

def deterministic_answer(query, data):
    q = query.lower()
    if "total spending" in q:
        for cust in ['Amit', 'Riya', 'Karan']:
            if cust.lower() in q:
                total = sum(tr['amount'] for tr in data if tr['customer'].lower() == cust.lower())
                return f"{cust} spent a total of ₹{total}."
    if "purchase history" in q:
        for cust in ['Amit', 'Riya', 'Karan']:
            if cust.lower() in q:
                purchases = [f"{tr['product']} for ₹{tr['amount']} on {tr['date']}" for tr in data if tr['customer'].lower()==cust.lower()]
                return f"{cust}'s purchases: " + ", ".join(purchases)
    if "transactions in" in q or "monthly transaction" in q:
        for month in ['2024-01','2024-02','2024-03']:
            if month in q:
                trans = [transaction_sentence(tr) for tr in data if tr['date'].startswith(month)]
                return f"Transactions in {month}: " + "; ".join(trans)
    if "average transaction" in q:
        amounts = [tr['amount'] for tr in data]
        avg = round(sum(amounts)/len(amounts),2)
        return f"Average transaction amount is ₹{avg}."
    if "most often" in q or "most common product" in q or "most frequently purchased product" in q:
        from collections import Counter
        prod = Counter(tr['product'] for tr in data).most_common(1)[0][0]
        return f"Product purchased most often is {prod}."
    return None

def main():
    print("Welcome to the Transactional RAG Chatbot!")
    data = load_transactions()
    texts = preprocess_transactions(data)
    vectorizer, embeddings = create_embeddings(texts)
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        results = retrieve_transactions(query, embeddings, texts, top_k=3, vectorizer=vectorizer)
        max_sim = results[0][1] if results else 0
        print("-- Top retrieved --")
        for text, sim in results:
            print(f'"{text}"   [score: {sim:.2f}]')
        if max_sim < 0.25:
            print("I don't have enough information in the uploaded documents.")
            continue
        answer = deterministic_answer(query, data)
        if answer:
            print("Answer:", answer)
        else:
            print("I don't have enough information in the uploaded documents.")

if __name__=="__main__":
    main()
