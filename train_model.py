import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocess import preprocess_pipeline
import os

def load_sms_data(filepath):
    """Load SMS Spam Collection dataset"""
    print("Loading SMS data...")
    sms_df = pd.read_csv(filepath, sep='\t', names=['label', 'text'])
    sms_df['label'] = sms_df['label'].map({'ham': 0, 'spam': 1})
    print(f"SMS data loaded: {len(sms_df)} samples")
    print(f"SMS distribution - Ham: {len(sms_df[sms_df['label']==0])}, Spam: {len(sms_df[sms_df['label']==1])}")
    return sms_df

def load_email_data(filepath):
    """Load Email dataset and label based on file path"""
    print("Loading Email data...")
    try:
        # Try reading with different encoding options
        emails_df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        emails_df = pd.read_csv(filepath, encoding='latin-1')
    except PermissionError:
        print("Permission error. Trying to copy file first...")
        import shutil
        shutil.copy2(filepath, 'emails_temp.csv')
        emails_df = pd.read_csv('emails_temp.csv', encoding='utf-8')
        os.remove('emails_temp.csv')
    
    # Label based on file path: if 'junk' or 'spam' in path, it's spam (1), else ham (0)
    emails_df['label'] = emails_df['file'].apply(
        lambda x: 1 if 'junk' in str(x).lower() or 'spam' in str(x).lower() else 0
    )
    emails_df = emails_df.rename(columns={'message': 'text'})
    
    # Sample if dataset is too large (optional - comment out if you want full dataset)
    if len(emails_df) > 10000:
        print(f"Sampling emails from {len(emails_df)} to 10000 for faster training...")
        emails_df = emails_df.sample(n=10000, random_state=42)
    
    print(f"Email data loaded: {len(emails_df)} samples")
    print(f"Email distribution - Ham: {len(emails_df[emails_df['label']==0])}, Spam: {len(emails_df[emails_df['label']==1])}")
    return emails_df[['text', 'label']]

def preprocess_data(df):
    """Apply preprocessing pipeline to all text"""
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_pipeline)
    print("Preprocessing complete!")
    return df

def train_model():
    """Main training function"""
    print("="*60)
    print("UNIVERSAL SPAM DETECTOR - MODEL TRAINING")
    print("="*60)
    
    # Step 1: Load datasets
    sms_df = load_sms_data('data/SMSSpamCollection')
    email_df = load_email_data('data/emails.csv')
    
    # Step 2: Merge datasets
    print("\nMerging datasets...")
    combined_df = pd.concat([sms_df[['text', 'label']], email_df[['text', 'label']]], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")
    print(f"Combined distribution - Ham: {len(combined_df[combined_df['label']==0])}, Spam: {len(combined_df[combined_df['label']==1])}")
    
    # Step 3: Preprocess
    combined_df = preprocess_data(combined_df)
    
    # Remove empty texts after preprocessing
    combined_df = combined_df[combined_df['clean_text'].str.len() > 0]
    
    # Step 4: Split data
    X = combined_df['clean_text']
    y = combined_df['label']
    
    print(f"\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Step 5: TF-IDF Vectorization
    print("\nApplying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Step 6: Train Logistic Regression
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train_tfidf, y_train)
    print("Training complete!")
    
    # Step 7: Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Step 8: Save model and vectorizer
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    print("✓ Model saved to model/model.pkl")
    print("✓ Vectorizer saved to model/vectorizer.pkl")
    print("\nTraining complete! You can now run the Flask app with: python app.py")

if __name__ == '__main__':
    train_model()
