import os
import re
import string
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("transformation_error.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lower_case(text):
    """Convert text to lowercase."""
    return str(text).lower()

def remove_urls(text):
    """Remove URLs."""
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))

def remove_html(text):
    """Remove HTML tags."""
    return re.sub(r"<.*?>", "", str(text))

def remove_emails(text):
    """Remove email addresses."""
    return re.sub(r"\S+@\S+", "", str(text))

def remove_mentions(text):
    """Remove Twitter mentions."""
    return re.sub(r"@\w+", "", str(text))

def remove_hashtags(text):
    """Remove only # symbol while keeping hashtag word."""
    return re.sub(r"#", "", str(text))

def remove_numbers(text):
    """Remove digits."""
    return re.sub(r"\d+", "", str(text))

def remove_punctuation(text):
    """Remove punctuation."""
    text = re.sub(f"[{re.escape(string.punctuation)}]"," ",str(text))
    return re.sub(r"\s+", " ", text).strip()

def remove_stop_words(text):
    """Remove English stopwords."""
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def lemmatization(text):
    """Perform lemmatization."""
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def remove_extra_spaces(text):
    """Remove extra spaces."""
    return re.sub(r"\s+", " ", text).strip()

def remove_small_sentences(df):
    """
    Remove tweets containing fewer than three words.
    """
    df = df.copy()
    df = df[df["content"].str.split().str.len() >= 3]
    return df

def normalize_text(df):
    """
    Perform complete preprocessing on tweet text.
    NOTE:
    Sentiment labels are NOT modified.
    Only the 'content' column is cleaned.
    """
    try:
        df = df.copy()

        # Handle missing values
        df["content"] = (df["content"].fillna("").astype(str))
        logger.info("Starting text preprocessing...")
        df["content"] = df["content"].apply(lower_case)
        logger.debug("Lowercase completed.")
        df["content"] = df["content"].apply(remove_urls)
        logger.debug("URLs removed.")
        df["content"] = df["content"].apply(remove_html)
        logger.debug("HTML tags removed.")
        df["content"] = df["content"].apply(remove_emails)
        logger.debug("Emails removed.")
        df["content"] = df["content"].apply(remove_mentions)
        logger.debug("Twitter mentions removed.")
        df["content"] = df["content"].apply(remove_hashtags)
        logger.debug("Hashtags cleaned.")
        df["content"] = df["content"].apply(remove_numbers)
        logger.debug("Numbers removed.")
        df["content"] = df["content"].apply(remove_punctuation)
        logger.debug("Punctuation removed.")
        df["content"] = df["content"].apply(remove_extra_spaces)
        df["content"] = df["content"].apply(remove_stop_words)
        logger.debug("Stopwords removed.")
        df["content"] = df["content"].apply(lemmatization)
        logger.debug("Lemmatization completed.")
        df = remove_small_sentences(df)
        logger.info("Text preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error("Error during preprocessing: %s",e)
        raise

def save_processed_data(train_df, test_df, output_path):
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path,"train_processed.csv"),index=False)
    test_df.to_csv(os.path.join(output_path,"test_processed.csv"),index=False)
    logger.info("Processed datasets saved to %s",output_path)

def main():
    try:
        logger.info("Loading train and test datasets...")
        train_df = pd.read_csv("./data/raw/train.csv")
        test_df = pd.read_csv("./data/raw/test.csv")
        logger.info("Train Shape: %s",train_df.shape)
        logger.info("Test Shape: %s",test_df.shape)
        # Normalize text
        train_df = normalize_text(train_df)
        test_df = normalize_text(test_df)
        logger.info("Processed Train Shape: %s",train_df.shape)
        logger.info("Processed Test Shape: %s",test_df.shape)
        # Save processed datasets
        save_processed_data(train_df,test_df,"./data/interim")
        logger.info("Data Transformation Completed Successfully.")
    except Exception as e:
        logger.error("Pipeline Failed: %s",e)
        print(e)

if __name__ == "__main__":
    main()