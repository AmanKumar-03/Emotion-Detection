import os
import re
import string
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

os.makedirs(
    "logs",
    exist_ok=True
)
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

# =====================================================
# NLTK Setup
# =====================================================

def download_nltk():

    packages = [
        "stopwords",
        "wordnet",
        "punkt"
    ]


    for package in packages:

        try:

            nltk.data.find(
                f"corpora/{package}"
            )


        except LookupError:

            nltk.download(
                package,
                quiet=True
            )



download_nltk()



lemmatizer = WordNetLemmatizer()


stop_words = set(
    stopwords.words("english")
)

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

    try:

        df = df.copy()


        df["content"] = (
            df["content"]
            .fillna("")
            .astype(str)
        )


        steps = [

            lower_case,

            remove_urls,

            remove_html,

            remove_emails,

            remove_mentions,

            remove_hashtags,

            remove_numbers,

            remove_punctuation,

            remove_extra_spaces,

            remove_stop_words,

            lemmatization

        ]


        for step in steps:

            df["content"] = (
                df["content"]
                .apply(step)
            )


        df = remove_small_sentences(
            df
        )


        return df



    except Exception as e:

        logger.exception(
            "Preprocessing failed: %s",
            e
        )

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