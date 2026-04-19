# Universal Spam Detector

A web application that detects spam in SMS messages and emails using machine learning.

## Features

- 🤖 ML-powered spam detection
- 📱 SMS and email support
- 🎯 Real-time prediction with confidence scores
- 🌐 Web interface built with Flask

## Tech Stack

- **Backend**: Flask (Python)
- **ML Model**: Scikit-learn (TF-IDF + Classifier)
- **NLP**: NLTK for text preprocessing
- **Frontend**: HTML/CSS/JavaScript

## Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/muhammadarmeen/spam-detector.git
cd spam-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the app**
Open your browser and visit: `http://localhost:5000`

## Deploy to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel
```

Or deploy directly from the Vercel dashboard by connecting your GitHub repository.

## Project Structure

```
spam-detector/
├── app.py                 # Main Flask application
├── preprocess.py          # Text preprocessing pipeline
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment config
├── model/                # Trained ML models
│   ├── model.pkl
│   └── vectorizer.pkl
├── templates/            # HTML templates
│   ├── index.html
│   └── result.html
└── data/                 # Training datasets
    └── SMSSpamCollection
```

## API Endpoints

- `GET /` - Home page with input form
- `POST /predict` - Predict if text is spam

## How It Works

1. **Text Preprocessing**: Clean text, remove stopwords, lemmatization
2. **Feature Extraction**: Convert text to TF-IDF features
3. **Prediction**: ML classifies as spam or ham
4. **Results**: Display prediction with confidence score

## License

MIT License

## Author

Muhammad Armeen
