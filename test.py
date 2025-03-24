import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import os

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load the saved model and preprocessing tools
def load_resources(model_path, tokenizer_path, vectorizer_path):
    """Load the saved model, tokenizer and TF-IDF vectorizer"""
    
    # Load the model
    model = load_model(model_path)
    
    # Load the tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load the TF-IDF vectorizer
    with open(vectorizer_path, 'rb') as handle:
        tfidf_vectorizer = pickle.load(handle)
    
    return model, tokenizer, tfidf_vectorizer

# Text preprocessing functions
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    # Replace numbers with 'NUM'
    text = re.sub(r'\d+', ' NUM ', text)
    # Remove special characters but keep punctuation for sentiment analysis
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenize text and remove stopwords while keeping important words"""
    tokens = word_tokenize(text)
    # Get stopwords but keep important words that might be relevant for fake news detection
    stop_words = set(stopwords.words('english'))
    important_words = {'not', 'no', 'nor', 'but', 'however', 'although', 'though'}
    filtered_stop_words = stop_words - important_words
    tokens = [word for word in tokens if word not in filtered_stop_words]
    return ' '.join(tokens)

def extract_features(text):
    """Extract additional features from text for model input"""
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Sentiment features
    sentiment = sia.polarity_scores(text)
    
    # Readability features
    readability = textstat.flesch_reading_ease(text)
    grade_level = textstat.flesch_kincaid_grade(text)
    
    # Text statistics
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
    
    # Special character ratio
    special_char_count = len(re.findall(r'[.,!?;:]', text))
    special_char_ratio = special_char_count / max(1, len(text))
    
    # Create feature dictionary
    features = {
        'sentiment_pos': sentiment['pos'],
        'sentiment_neg': sentiment['neg'],
        'sentiment_neu': sentiment['neu'],
        'sentiment_compound': sentiment['compound'],
        'readability': readability,
        'grade_level': grade_level,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'special_char_ratio': special_char_ratio
    }
    
    return features

# Prediction function
def predict_fake_news(title, text, model, tokenizer, tfidf_vectorizer, max_length=300):
    """
    Predict if a news article is fake or real
    
    Parameters:
    title (str): The title of the news article
    text (str): The text content of the news article
    model: The loaded Keras model
    tokenizer: The loaded tokenizer
    tfidf_vectorizer: The loaded TF-IDF vectorizer
    max_length (int): Maximum sequence length used during training
    
    Returns:
    float: Probability of the article being real (0 = fake, 1 = real)
    str: Classification label
    dict: Additional information about the prediction
    """
    # Combine title and text
    content = f"{title} [SEP] {text}"
    
    # Clean the text
    cleaned_content = clean_text(content)
    
    # Tokenize and remove stopwords
    processed_content = tokenize_and_remove_stopwords(cleaned_content)
    
    # Extract features for the model
    features = extract_features(cleaned_content)
    features_df = pd.DataFrame([features])
    
    # Create sequence input
    sequences = tokenizer.texts_to_sequences([processed_content])
    X_seq = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Create TF-IDF features
    X_tfidf = tfidf_vectorizer.transform([processed_content]).toarray()
    
    # Combine all features
    X_features = np.hstack((X_tfidf, features_df.values))
    
    # Make prediction
    prediction = model.predict([X_seq, X_features])[0][0]
    
    # Determine classification and confidence
    label = "REAL" if prediction >= 0.5 else "FAKE"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    # Prepare additional info
    sentiment_score = features['sentiment_compound']
    sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
    
    info = {
        'prediction_probability': float(prediction),
        'confidence': float(confidence),
        'sentiment': sentiment_label,
        'sentiment_score': sentiment_score,
        'readability_score': features['readability'],
        'grade_level': features['grade_level'],
    }
    
    return prediction, label, info

# Function to print detailed prediction results
def print_prediction_results(title, prediction, label, info):
    """Print detailed prediction results"""
    print("\n" + "="*50)
    print(f"TITLE: {title}")
    print("="*50)
    print(f"\nPREDICTION: {label} NEWS")
    print(f"Confidence: {info['confidence']*100:.2f}%")
    print(f"\nAdditional Information:")
    print(f"- Raw prediction score: {info['prediction_probability']:.4f}")
    print(f"- Sentiment: {info['sentiment']} ({info['sentiment_score']:.2f})")
    print(f"- Readability: {info['readability_score']:.1f} (Flesch Reading Ease)")
    print(f"- Grade Level: {info['grade_level']:.1f} (Flesch-Kincaid)")
    print("="*50)

# Main function to load model and make predictions
def main():
    # Define file paths here in the main function
    model_path = 'best_fake_news_detection_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    
    # Load the model and preprocessing tools
    print("Loading model and resources...")
    try:
        model, tokenizer, tfidf_vectorizer = load_resources(model_path, tokenizer_path, vectorizer_path)
        print("Model and resources loaded successfully!")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return
    
    # Sample news articles for testing
    sample_articles = [
        {
            'title': 'Scientists discover new renewable energy source',
            'text': 'Researchers at MIT have announced a breakthrough in renewable energy technology. The new method harnesses ambient thermal energy with twice the efficiency of current solar panels. The research team published their findings in Nature Energy after three years of laboratory testing. Industry experts call this discovery a potential game-changer for clean energy production. The technology could be commercially viable within five years, according to lead researcher Dr. Sarah Chen.'
        },
        {
            'title': 'As U.S. budget fight looms, Republicans flip their fiscal script","WASHINGTON (Reuters) ',
            'text': """he head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary” spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump) administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” Meadows, chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats are saying that’s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I don’t see where the rationale is. ... Eventually you run out of other people’s money,” he said. Meadows was among Republicans who voted in late December for their party’s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. “It’s interesting to hear Mark talk about fiscal responsibility,” Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. “This is one of the least ... fiscally responsible bills we’ve ever seen passed in the history of the House of Representatives. I think we’re going to be paying for this for many, many years to come,” Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or “entitlement reform,” as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, “entitlement” programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryan’s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the “Dreamers,” people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. “We need to do DACA clean,” she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid. """
        }
    ]
    
    # Make predictions for sample articles
    print("\nAnalyzing sample articles...")
    
    for i, article in enumerate(sample_articles):
        prediction, label, info = predict_fake_news(
            article['title'], 
            article['text'], 
            model, 
            tokenizer, 
            tfidf_vectorizer
        )
        
        print_prediction_results(article['title'], prediction, label, info)

if __name__ == "__main__":
    main()