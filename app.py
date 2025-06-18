import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("products.csv")  # Ensure ProductName, Category, Price, ImageURL

# TF-IDF model
vectorizer = TfidfVectorizer()
product_vectors = vectorizer.fit_transform(df['ProductName'])

# Recommend function
def recommend(product_name, top_n=5):
    user_vec = vectorizer.transform([product_name])
    similarity = cosine_similarity(user_vec, product_vectors)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# UI
st.set_page_config(page_title="Smart Product Recommender", layout="wide")
st.markdown("<h1 style='color:white;'>🛍️ Smart Product Recommender System</h1>", unsafe_allow_html=True)

suggestions = ["headphones", "shoes", "laptop", "smartphone", "backpack", "camera"]
input_box = st.empty()
user_input = ""

# Animate rotating placeholders
for i, word in enumerate(suggestions):
    key = f"dynamic_input_{i}"  # Unique key each loop
    user_input = input_box.text_input(
        "Search for a product", 
        placeholder=f"Try typing '{word}'", 
        key=key
    )
    time.sleep(1)
    if user_input:
        break

# After typing
if user_input:
    with st.spinner("🔍 Finding recommendations..."):
        results = recommend(user_input)
        st.success("🎯 Top Recommendations:")

        for _, row in results.iterrows():
            st.markdown("---")
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(row['ImageURL'], width=150, caption=row['ProductName'])
            with cols[1]:
                st.markdown(f"**🛍️ Product:** {row['ProductName']}")
                st.markdown(f"**📦 Category:** {row['Category']}")
                st.markdown(f"**💰 Price:** ₹{float(row['Price']):,.2f}")
