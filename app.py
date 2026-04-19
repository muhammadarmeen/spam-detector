from flask import Flask, render_template, request
import joblib
from preprocess import preprocess_pipeline

app = Flask(__name__)

# Load model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
print("Model loaded successfully!")

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the input text is spam or ham"""
    try:
        # Get text from form
        text = request.form['text']
        
        if not text or text.strip() == '':
            return render_template('index.html', error='Please enter some text to analyze.')
        
        # Preprocess text
        processed_text = preprocess_pipeline(text)
        
        # Vectorize
        text_tfidf = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        # Get result
        result = 'SPAM' if prediction == 1 else 'HAM (Not Spam)'
        confidence = max(probability) * 100
        spam_probability = probability[1] * 100 if len(probability) > 1 else 0
        ham_probability = probability[0] * 100
        
        return render_template('result.html', 
                             result=result, 
                             confidence=confidence,
                             spam_probability=spam_probability,
                             ham_probability=ham_probability,
                             original_text=text)
    
    except Exception as e:
        return render_template('index.html', error=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    print("="*60)
    print("UNIVERSAL SPAM DETECTOR - FLASK APP")
    print("="*60)
    print("Starting server...")
    print("Access the app at: http://localhost:5000")
    print("Press CTRL+C to quit")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
