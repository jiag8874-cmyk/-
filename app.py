import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 학습 데이터
# -------------------------------
data = {
    "text": [
        "The government announced a new education policy.",
        "Scientists discovered water on Mars.",
        "Researchers developed a new cancer treatment.",
        "A university launched a scholarship program.",
        "New renewable energy technology was introduced.",

        "Aliens landed in Seoul secretly.",
        "Drinking lemon juice cures all diseases instantly.",
        "Celebrity is actually a robot created by NASA.",
        "Invisible cars will be sold next month.",
        "Students no longer need sleep because of brain chip."
    ],
    "label": [1,1,1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# 의심 단어
# -------------------------------
suspicious_words = [
    "secret", "shocking", "miracle", "guaranteed",
    "instant", "100%", "breaking", "unbelievable",
    "cure", "hidden", "exclusive"
]

# 신뢰도 낮은 사이트 예시
untrusted_domains = [
    "fake-news.com",
    "clickbait-news.net",
    "viral-shocking.xyz"
]

# -------------------------------
# 분석 함수
# -------------------------------
def analyze_text(text):
    score = 0
    reasons = []

    lower_text = text.lower()

    # 1. 의심 단어 검사
    found_words = []
    for word in suspicious_words:
        if word in lower_text:
            found_words.append(word)

    if found_words:
        score += len(found_words) * 10
        reasons.append(f"자극적인 단어 발견: {', '.join(found_words)}")

    # 2. 느낌표 개수
    exclamation_count = text.count("!")
    if exclamation_count >= 3:
        score += 15
        reasons.append(f"느낌표 과다 사용 ({exclamation_count}개)")

    # 3. 대문자 비율
    uppercase_count = sum(1 for c in text if c.isupper())
    if len(text) > 0:
        upper_ratio = uppercase_count / len(text)
        if upper_ratio > 0.2:
            score += 15
            reasons.append("대문자 과다 사용")

    # 4. 반복 단어 검사
    words = re.findall(r'\b\w+\b', lower_text)
    repeated = set([word for word in words if words.count(word) >= 4])
    if repeated:
        score += 10
        reasons.append(f"반복 단어 발견: {', '.join(repeated)}")

    return score, reasons


def analyze_url(url):
    score = 0
    reasons = []

    if url:
        try:
            domain = urlparse(url).netloc.lower()

            if domain in untrusted_domains:
                score += 30
                reasons.append(f"신뢰도 낮은 사이트: {domain}")

            if ".xyz" in domain or ".click" in domain:
                score += 15
                reasons.append("의심스러운 도메인 사용")

        except:
            reasons.append("URL 분석 실패")

    return score, reasons


# -------------------------------
# UI
# -------------------------------
st.title("AI 가짜뉴스 판별 앱")
st.write("텍스트 특징 + AI 분석을 통해 가짜뉴스 가능성을 평가한다.")

news_text = st.text_area("뉴스 기사 입력")
news_url = st.text_input("기사 URL 입력 (선택)")

if st.button("분석하기"):
    if news_text.strip() == "":
        st.warning("뉴스 내용을 입력하세요.")
    else:
        total_score = 0
        all_reasons = []

        # 규칙 기반 분석
        text_score, text_reasons = analyze_text(news_text)
        total_score += text_score
        all_reasons.extend(text_reasons)

        url_score, url_reasons = analyze_url(news_url)
        total_score += url_score
        all_reasons.extend(url_reasons)

        # AI 분석
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0]

        fake_prob = probability[0] * 100
        real_prob = probability[1] * 100

        st.subheader("AI 분석 결과")
        st.write(f"가짜뉴스 확률: {fake_prob:.2f}%")
        st.write(f"진짜뉴스 확률: {real_prob:.2f}%")

        st.subheader("규칙 기반 분석")
        st.write(f"의심 점수: {total_score}점")

        if all_reasons:
            for reason in all_reasons:
                st.write("- " + reason)
        else:
            st.write("특별한 의심 요소 없음")

        final_score = total_score + fake_prob * 0.3

        st.subheader("최종 판별")
        if final_score >= 40:
            st.error("가짜뉴스일 가능성이 높다")
        else:
            st.success("상대적으로 신뢰 가능성이 높다")
