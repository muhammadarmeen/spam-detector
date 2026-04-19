import streamlit as st
import joblib
from preprocess import preprocess_pipeline
import os

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load ML model and vectorizer with caching"""
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Load models
model, vectorizer = load_model()

# App UI
st.title("🛡️ Universal Spam Detector")
st.markdown("Enter a message below to check if it's **spam** or **not spam (ham)**.")

# Text input
user_input = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
    height=150
)

# Predict button
if st.button("🔍 Check for Spam", type="primary", use_container_width=True):
    if not user_input or user_input.strip() == '':
        st.error("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your message..."):
            try:
                # Preprocess text
                processed_text = preprocess_pipeline(user_input)
                
                # Vectorize
                text_tfidf = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(text_tfidf)[0]
                probability = model.predict_proba(text_tfidf)[0]
                
                # Get results
                is_spam = prediction == 1
                confidence = max(probability) * 100
                spam_probability = probability[1] * 100 if len(probability) > 1 else 0
                ham_probability = probability[0] * 100
                
                # Display result
                st.divider()
                if is_spam:
                    st.error("🚨 **SPAM DETECTED!**")
                    st.warning("This message appears to be spam.")
                else:
                    st.success("✅ **SAFE MESSAGE**")
                    st.info("This message appears to be legitimate (ham).")
                
                # Display confidence metrics
                st.divider()
                st.subheader("📊 Analysis Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col2:
                    st.metric("Classification", "SPAM" if is_spam else "HAM")
                
                # Probability bars
                st.write("**Spam Probability:**")
                st.progress(spam_probability / 100)
                st.write(f"{spam_probability:.1f}%")
                
                st.write("**Ham Probability:**")
                st.progress(ham_probability / 100)
                st.write(f"{ham_probability:.1f}%")
                
                # Original text
                st.divider()
                st.subheader("📝 Original Message")
                st.text_area("", value=user_input, height=100, disabled=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses machine learning to detect spam messages.
    
    **How it works:**
    1. Text preprocessing (cleaning, lemmatization)
    2. Feature extraction (TF-IDF)
    3. ML classification
    
    **Built with:**
    - Streamlit
    - Scikit-learn
    - NLTK
    """)
    
    st.divider()
    st.write("**Examples to try:**")
    st.markdown("""
    🚨 *Spam:*
    - "You won $1000! Click here to claim"
    - "URGENT: Your account has been compromised"
    
    ✅ *Safe:*
    - "Hey, are we still meeting for lunch?"
    - "The project deadline is next Friday"
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with ❤️ by Muhammad Armeen | 
        <a href='https://github.com/muhammadarmeen/spam-detector'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
