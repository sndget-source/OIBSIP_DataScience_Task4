# Task 4 — Email Spam Detection with Machine Learning

**Internship:** Oasis Infobyte Data Science Internship (OIBSIP)

## What this project is about
For this task, I built an NLP model that can tell whether a text message is spam or a normal (ham) message, just based on the message content. This is basically the same idea behind spam filters in email and SMS apps, so I wanted to actually build one from scratch and see how well a fairly simple model can do at it.

## Dataset
I used the **SMS Spam Collection dataset** from Kaggle/UCI (`spam.csv`). It has around 5,500 text messages, each labeled as either "spam" or "ham," pulled together from a mix of real SMS sources.

## What I did
- Loaded the dataset and cleaned up the raw file (it comes with some extra junk columns and a slightly odd encoding).
- Checked the class distribution first — turns out the dataset is pretty imbalanced, with far more ham messages than spam (roughly 87% ham vs 13% spam), which matches real life since most texts people get aren't spam.
- Cleaned the message text: lowercased everything, stripped out punctuation, and removed common stopwords so the model isn't wasting effort on filler words like "the" or "and".
- Converted the cleaned text into numeric features using **TF-IDF**, which scores each word based on how important it is to a specific message relative to the whole dataset, instead of just counting raw word frequency.
- Split the data into training and test sets.
- Trained two models: **Multinomial Naive Bayes** (the standard go-to for text classification) and **SVM** as a second model to compare against it.
- Evaluated both using accuracy, precision, recall, F1-score, and confusion matrices.
- Generated wordclouds for spam vs ham messages just to visually confirm the model has real patterns to learn from.

## Why recall matters here
I put some thought into this specifically because the checklist asked for it: for spam detection, I'd argue recall matters more than precision. Recall tells you how much actual spam the model successfully caught — if it's low, real spam is slipping through into the inbox, which defeats the purpose of the filter. Precision still matters since you don't want real messages wrongly marked as spam, but missing spam altogether feels like the bigger practical failure, especially since most spam folders are easy to double check anyway.

## Results
*(Fill in after running: which model — Naive Bayes or SVM — performed better, and the actual accuracy/precision/recall/F1 numbers for each. Also note anything you noticed from the wordclouds, like specific spam trigger words that showed up — free, win, claim, prize, txt are common ones for this dataset.)*

## Tech Stack
Python, pandas, scikit-learn (TF-IDF, Naive Bayes, SVM), NLTK, WordCloud, Jupyter Notebook

## Files
- `Shalini Naga Dhonthu Balla_Task4.ipynb` — the full notebook, from raw text to trained, evaluated models.

## How to run it
1. Download `spam.csv` from Kaggle ("SMS Spam Collection Dataset") and place it in the same folder as the notebook.
2. Install the required libraries: `pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud`
3. Open the notebook and run all cells top to bottom.
