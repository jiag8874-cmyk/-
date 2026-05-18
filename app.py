import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "text": [
        "The government announced a new education policy.",
        "Scientists discovered water on Mars.",
        "Aliens landed in Seoul secretly."
    ],
    "label": [1, 1, 0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

st.title("가짜뉴스 판별 앱")

news = st.text_area("뉴스 입력")

if st.button("판별"):
    result = model.predict(vectorizer.transform([news]))[0]
    if result == 1:
        st.success("진짜 뉴스일 가능성이 높다")
    else:
        st.error("가짜 뉴스일 가능성이 높다")
