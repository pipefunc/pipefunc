---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: pipefunc
  language: python
  name: python3
---

# Natural Language Processing Pipeline for Text Summarization

```{try-notebook}
```

```{note}
This example uses `nltk` for text processing. If you don't have it installed, you can install it using `pip install nltk`.
```

Let's create a simplified NLP workflow for text summarization with steps involving tokenization, keyword extraction, summary generation, and sentiment analysis. This example will demonstrate the use of `pipefunc` to handle dependencies and illustrate `mapspec` by processing multiple texts.

```{code-cell} ipython3
import nltk
import numpy as np
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

from pipefunc import Pipeline, pipefunc

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# Step 1: Text Tokenization
@pipefunc(output_name="tokens", mapspec="text[n] -> tokens[n]")
def tokenize_text(text):
    from nltk.corpus import stopwords

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words


# Step 2: Keyword Extraction
@pipefunc(output_name="keywords", mapspec="tokens[n] -> keywords[n]")
def extract_keywords(tokens):
    freq_dist = FreqDist(tokens)
    common_keywords = freq_dist.most_common(5)
    return [word for word, _ in common_keywords]


# Step 3: Summary Generation
@pipefunc(output_name="summary", mapspec="text[n], keywords[n] -> summary[n]")
def generate_summary(text, keywords):
    sentences = sent_tokenize(text)
    important_sentences = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in keywords)
    ]
    return " ".join(important_sentences[:2])  # Return the first two important sentences


# Step 4: Sentiment Analysis
@pipefunc(output_name="sentiment", mapspec="summary[n] -> sentiment[n]")
def analyze_sentiment(summary):
    # Simplified sentiment analysis: More positive words = Positive sentiment
    positive_words = {"good", "great", "excellent", "positive", "fortunate"}
    negative_words = {"bad", "terrible", "poor", "negative", "unfortunate"}
    words = set(summary.lower().split())
    sentiment_score = len(words & positive_words) - len(words & negative_words)
    if sentiment_score > 0:
        return "Positive"
    if sentiment_score < 0:
        return "Negative"
    return "Neutral"


# Step 5: Summarization Result Aggregation
@pipefunc(output_name="result_summary")
def aggregate_summarization(sentiment):
    # Convert the sentiment masked array to a list
    sentiment_list = np.array(sentiment).tolist()

    # Count occurrences of each sentiment type
    positive_count = sentiment_list.count("Positive")
    negative_count = sentiment_list.count("Negative")
    neutral_count = sentiment_list.count("Neutral")

    return {"Positive": positive_count, "Negative": negative_count, "Neutral": neutral_count}


# Create the pipeline
pipeline_sentiment = Pipeline(
    [
        tokenize_text,
        extract_keywords,
        generate_summary,
        analyze_sentiment,
        aggregate_summarization,
    ],
)

# Example texts to summarize
texts = [
    "The movie was excellent! The performances were outstanding, and the plot was captivating.",
    "The movie was bad and boring. I found it dull and slow with no gripping moments.",
    "An alright film with a good sense of humor but lacking depth in character development.",
]

# Run the pipeline on texts
results_summary = pipeline_sentiment.map({"text": texts}, parallel=True)
print("Summarization Sentiment Summary:", results_summary["result_summary"].output)
```

**Explanation:**

- **Text Tokenization (`tokenize_text`)**: Tokenizes each text, removing stop words using a `mapspec` for independent processing.
- **Keyword Extraction (`extract_keywords`)**: Extracts the most frequent words as keywords for each tokenized text.
- **Summary Generation (`generate_summary`)**: Generates a summary by selecting sentences with the most important keywords.
- **Sentiment Analysis (`analyze_sentiment`)**: Analyzes sentiment of summary texts using a set of positive and negative keywords for simplicity.
- **Result Aggregation (`aggregate_summarization`)**: Aggregates sentiments to provide an overall sentiment summary across all texts.

This example showcases the capability of `pipefunc` to manage complex workflows in NLP applications, leveraging `mapspec` to process multiple text inputs efficiently.
