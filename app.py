import streamlit as st
import pickle
import re

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Fake News Detection Framework",
    page_icon="📰",
    layout="wide"
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# CUSTOM DASHBOARD STYLE
# =========================================================

st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}


/* ---------- HEADER ---------- */

.hero {
    padding: 30px 32px;
    border-radius: 20px;
    margin-bottom: 22px;

    background: linear-gradient(
        135deg,
        #312e81 0%,
        #4f46e5 55%,
        #7c3aed 100%
    );

    color: white;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.20);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 7px;
    opacity: 0.92;
}


/* ---------- STAT CARDS ---------- */

.stat-card {
    padding: 20px;
    border-radius: 17px;
    text-align: center;
    min-height: 105px;

    background: white;
    border: 1px solid #e0e7ff;

    box-shadow: 0 5px 18px rgba(79, 70, 229, 0.08);

    transition: 0.2s;
}

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.15);
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    color: #4338ca;
}

.stat-label {
    font-size: 14px;
    color: #64748b;
    margin-top: 5px;
}


/* ---------- SECTION TITLES ---------- */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #1e1b4b;
    margin-top: 20px;
    margin-bottom: 12px;
}


/* ---------- INPUT AREA ---------- */

textarea {
    border-radius: 14px !important;
    border: 2px solid #c7d2fe !important;
}

textarea:focus {
    border-color: #6366f1 !important;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );

    color: white;
    border: none;
    border-radius: 12px;

    padding: 12px 20px;

    font-weight: 700;
    font-size: 16px;

    box-shadow: 0 6px 15px rgba(79, 70, 229, 0.25);

    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #4338ca,
        #6d28d9
    );

    transform: translateY(-2px);
}


/* ---------- PREDICTION CARD ---------- */

.prediction-card {
    padding: 30px 20px;
    border-radius: 18px;
    min-height: 290px;

    text-align: center;

    background: linear-gradient(
        145deg,
        #ffffff,
        #f5f3ff
    );

    border: 2px solid #c4b5fd;

    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.10);
}

.prediction-title {
    font-size: 31px;
    font-weight: 800;
    color: #4338ca;
}


/* ---------- PIPELINE ---------- */

.pipeline-card {
    padding: 18px 8px;

    border-radius: 15px;

    text-align: center;
    min-height: 135px;

    background: white;

    border: 1px solid #ddd6fe;

    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.07);

    transition: 0.2s;
}

.pipeline-card:hover {
    transform: translateY(-4px);
    border-color: #8b5cf6;
}


/* ---------- ALERTS ---------- */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: #ddd6fe;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    padding: 25px;
    margin-top: 30px;
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
📰 Fake News Detection Framework
</div>

<div class="hero-subtitle">
AI-Powered Text Classification for Detecting Fake and Genuine News
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# QUICK STATISTICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">44,898</div>
        <div class="stat-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">23,481</div>
        <div class="stat-label">Fake Articles</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">21,417</div>
        <div class="stat-label">Genuine Articles</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">2 Classes</div>
        <div class="stat-label">Binary Classification</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# =========================================================
# NEWS ANALYZER
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Analyze a News Article</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.55, 1])


# =========================================================
# INPUT PANEL
# =========================================================

with left:

    st.write("### 📝 News Article")

    news = st.text_area(
        "Paste article",
        height=290,
        placeholder="Paste the complete news article here...",
        label_visibility="collapsed"
    )

    analyze = st.button(
        "🔎 Analyze Article",
        use_container_width=True
    )


# =========================================================
# PREDICTION PANEL
# =========================================================

with right:

    st.write("### 🎯 Classification")

    if analyze:

        if not news.strip():

            st.warning("Please enter a news article.")

        else:

            cleaned_news = clean_text(news)

            news_vector = vectorizer.transform([cleaned_news])

            prediction = model.predict(news_vector)[0]

            probabilities = model.predict_proba(news_vector)[0]

            confidence = max(probabilities) * 100

            if prediction == 0:

                st.markdown(
                    f"""
                    <div class="prediction-card">

                    <div class="prediction-title">
                    🚨 FAKE NEWS
                    </div>

                    <br>

                    <p>
                    The model classified this article as
                    <b>Fake News</b>.
                    </p>

                    <br>

                    <h2>{confidence:.2f}%</h2>

                    <p>Model Confidence</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-card">

                    <div class="prediction-title">
                    ✅ GENUINE NEWS
                    </div>

                    <br>

                    <p>
                    The model classified this article as
                    <b>Genuine News</b>.
                    </p>

                    <br>

                    <h2>{confidence:.2f}%</h2>

                    <p>Model Confidence</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            st.progress(
                int(confidence),
                text=f"Confidence: {confidence:.2f}%"
            )

    else:

        st.info(
            "Enter a news article on the left and click "
            "**Analyze Article** to generate a prediction."
        )


# =========================================================
# DATASET ANALYSIS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Dataset Analysis</div>',
    unsafe_allow_html=True
)

chart_col, details_col = st.columns([1.6, 1])


with chart_col:

    st.write("#### Fake vs Genuine News")

    chart_data = {
        "Fake News": 23481,
        "Genuine News": 21417
    }

    st.bar_chart(chart_data)


with details_col:

    st.write("#### Dataset Summary")

    st.metric(
        "Total Articles",
        "44,898"
    )

    st.metric(
        "Fake Articles",
        "23,481"
    )

    st.metric(
        "Genuine Articles",
        "21,417"
    )


# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 Classification Pipeline</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4, p5 = st.columns(5)


with p1:

    st.markdown("""
    <div class="pipeline-card">

    <h2>📰</h2>

    <b>News Input</b>

    <br><br>

    Article entered

    </div>
    """, unsafe_allow_html=True)


with p2:

    st.markdown("""
    <div class="pipeline-card">

    <h2>🧹</h2>

    <b>Text Cleaning</b>

    <br><br>

    Text preprocessing

    </div>
    """, unsafe_allow_html=True)


with p3:

    st.markdown("""
    <div class="pipeline-card">

    <h2>🔢</h2>

    <b>TF-IDF</b>

    <br><br>

    Feature extraction

    </div>
    """, unsafe_allow_html=True)


with p4:

    st.markdown("""
    <div class="pipeline-card">

    <h2>🤖</h2>

    <b>Logistic Regression</b>

    <br><br>

    ML classification

    </div>
    """, unsafe_allow_html=True)


with p5:

    st.markdown("""
    <div class="pipeline-card">

    <h2>🎯</h2>

    <b>Prediction</b>

    <br><br>

    Fake or Genuine

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ Model Information</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.info("""
    🐍 **Language**

    Python
    """)


with m2:

    st.info("""
    🔢 **Feature Extraction**

    TF-IDF
    """)


with m3:

    st.info("""
    🤖 **Algorithm**

    Logistic Regression
    """)


with m4:

    st.info("""
    🎯 **Task**

    Binary Classification
    """)


# =========================================================
# RESPONSIBLE USE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">⚠️ Responsible Use</div>',
    unsafe_allow_html=True
)

st.warning(
    "This system classifies news text using patterns learned from "
    "the training dataset. Its output should be treated as an "
    "analytical prediction and not as independent verification "
    "of the factual truth of an article."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

📰 <b>Fake News Detection Framework</b>
<br>

Text Classification System using Machine Learning

<br>

PBL Project

</div>
""", unsafe_allow_html=True)