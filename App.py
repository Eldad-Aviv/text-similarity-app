import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

device = "cpu"  # Force CPU on Streamlit Cloud
model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

#model = SentenceTransformer('sentence-transformers/LaBSE')

st.title("Text Similarity Checker")

text1 = st.text_area("Enter the first text:", height=150)
text2 = st.text_area("Enter the second text:", height=150)

if st.button("Check Similarity"):
    if text1 and text2:
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        st.markdown(f"### Similarity Score: `{similarity:.2f}`")
    else:
        st.warning("Please enter both texts.")
