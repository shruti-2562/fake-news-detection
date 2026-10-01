import streamlit as st
import pickle
import re
import pandas as pd

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(
    page_title="Fake News Detection Framework",
    page_icon="📰",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL AND VECTORIZER
# -----------------------------------
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------------
# TEXT CLEANING
# -----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HEADER */
.header {
    background: linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb,
        #7c3aed
    );
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.25);
}

.header h1 {
    font-size: 36px;
    margin-bottom: 8px;
}

.header p {
    font-size: 17px;
    margin: 0;
}

/* SECTION TITLES */
.section {
    color: #1e3a8a;
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
    padding: 10px;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #6d28d9
    );
    color: white;
}

/* TEXT AREA */
textarea {
    border-radius: 12px !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 35px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div class="header">
    <h1>📰 Fake News Detection Framework</h1>
    <p>AI-Powered Text Classification for Detecting Fake and Genuine News</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------
# DATASET OVERVIEW
# -----------------------------------
st.markdown(
    '<div class="section">📌 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📚 Total Articles",
        "44,898"
    )

with col2:
    st.metric(
        "🚨 Fake Articles",
        "23,481"
    )

with col3:
    st.metric(
        "✅ Genuine Articles",
        "21,417"
    )

with col4:
    st.metric(
        "🔢 Classes",
        "2"
    )


# -----------------------------------
# NEWS ANALYZER
# -----------------------------------
st.divider()

st.markdown(
    '<div class="section">🔍 News Analyzer</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste a news article below and the trained machine learning "
    "model will classify it as Fake or Genuine."
)

news_text = st.text_area(
    "📝 News Article",
    height=230,
    placeholder="Paste the news article text here..."
)

# Example button
example_col1, example_col2 = st.columns([1, 5])

with example_col1:
    example_clicked = st.button(
        "💡 Example News"
    )

if example_clicked:
    st.info(
        "Example loaded. Copy this text into the News Article box "
        "and click Analyze News:"
    )
    st.code(
        "The government announced a new public health initiative "
        "to improve healthcare services across several regions. "
        "Officials stated that the program will focus on improving "
        "access to essential medical facilities and services."
    )

analyze = st.button(
    "🔎 Analyze News",
    use_container_width=True
)


# -----------------------------------
# PREDICTION
# -----------------------------------
if analyze:

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter some news text before analyzing."
        )

    else:

        cleaned = clean_text(news_text)

        transformed_text = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            transformed_text
        )[0]

        # Model probabilities
        probabilities = model.predict_proba(
            transformed_text
        )[0]

        confidence = float(
            max(probabilities)
        ) * 100

        fake_probability = float(
            probabilities[0]
        ) * 100

        genuine_probability = float(
            probabilities[1]
        ) * 100


        # -----------------------------------
        # TEXT STATISTICS
        # -----------------------------------
        word_count = len(news_text.split())
        character_count = len(news_text)
        sentence_count = len(
            re.findall(
                r'[.!?]+',
                news_text
            )
        )

        st.divider()

        st.subheader("📌 Article Statistics")

        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.metric(
                "📝 Words",
                word_count
            )

        with stat2:
            st.metric(
                "🔤 Characters",
                character_count
            )

        with stat3:
            st.metric(
                "📄 Sentences",
                sentence_count
            )


        # -----------------------------------
        # PREDICTION RESULT
        # -----------------------------------
        st.subheader(
            "📋 Prediction Result"
        )

        if prediction == 0:

            st.error(
                "🚨 FAKE NEWS DETECTED\n\n"
                "The model classified this article as Fake News."
            )

        else:

            st.success(
                "✅ GENUINE NEWS DETECTED\n\n"
                "The model classified this article as Genuine News."
            )


        # -----------------------------------
        # MODEL CONFIDENCE
        # -----------------------------------
        st.subheader(
            "🤖 Model Confidence"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )


        # -----------------------------------
        # PROBABILITY BREAKDOWN
        # -----------------------------------
        st.subheader(
            "📊 Prediction Probability"
        )

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:

            st.metric(
                "🚨 Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.progress(
                min(fake_probability / 100, 1.0)
            )

        with prob_col2:

            st.metric(
                "✅ Genuine Probability",
                f"{genuine_probability:.2f}%"
            )

            st.progress(
                min(genuine_probability / 100, 1.0)
            )

        st.caption(
            "These probabilities represent the classifier's estimated "
            "probability for each class. They do not guarantee that "
            "the article is factually true or false."
        )


# -----------------------------------
# DATASET ANALYSIS
# -----------------------------------
st.divider()

st.markdown(
    '<div class="section">📊 Dataset Analysis</div>',
    unsafe_allow_html=True
)

chart_col, info_col = st.columns(
    [1.7, 1]
)


# Chart data
chart_data = pd.DataFrame(
    {
        "News Type": [
            "Fake News",
            "Genuine News"
        ],
        "Articles": [
            23481,
            21417
        ]
    }
).set_index("News Type")


# -----------------------------------
# BAR CHART
# -----------------------------------
with chart_col:

    st.write(
        "### 📈 Fake vs Genuine News Distribution"
    )

    st.bar_chart(
        chart_data,
        use_container_width=True
    )


# -----------------------------------
# DATASET SHARE
# -----------------------------------
with info_col:

    st.write(
        "### 📊 Dataset Share"
    )

    st.metric(
        "🚨 Fake News Share",
        "52.3%"
    )

    st.metric(
        "✅ Genuine News Share",
        "47.7%"
    )

    st.info(
        "The dataset contains two classes with a relatively "
        "balanced distribution of fake and genuine news."
    )


# -----------------------------------
# CLASSIFICATION TASK
# -----------------------------------
st.divider()

st.markdown(
    '<div class="section">⚖️ Classification Task</div>',
    unsafe_allow_html=True
)

task1, task2 = st.columns(2)

with task1:

    st.success(
        "🎯 **Task: Binary Text Classification**\n\n"
        "The model predicts one of two classes: "
        "Fake News or Genuine News."
    )

with task2:

    st.info(
        "🧠 **Technique: TF-IDF + Logistic Regression**\n\n"
        "TF-IDF converts text into numerical features and "
        "Logistic Regression performs the classification."
    )


# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    '<div class="footer">'
    'Fake News Detection Framework | '
    'Machine Learning Based Text Classification'
    '</div>',
    unsafe_allow_html=True
)