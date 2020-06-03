# text-classifier
Scaffolding to create classifier based on natural language.

## Dependencies
This project needs matplotlib, seaborn, numpy, scikit-learn, and spacy. Additionally, it is necessary to download spacy's `en_core_web_md` pre-trainings ([see here for instructions](https://spacy.io/models/en)).

## Usage
First, obtain some data
```python
text, labels = load_data()
data = (text, labels)
```

Define a model with `sklearn`
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
```

Finally, train the model, evaluate it, use it for prediction
```python
trained_model, classes = classifier.train_model(logreg, training_data, spacy_tokens=True)
classifier.test_model(trained_model, training_data, classes)
classifier.predict_category('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', trained_model)
```

## Settings
There are three modes, mutually exclusive:
1. a fast, simple mode based on a tf-idf vectorizer and with a trivial tokenizer. 
2. a slower one using spacy's tokenizer
3. a slower one using Glove embeddings

