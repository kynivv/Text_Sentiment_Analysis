from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

text = "This pizza is awesome!"


# preprocess text
text_words = []

for word in text.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'

    elif word.startswith('http'):
        word = 'http'
    text_words.append(word)

text_proc = " ".join(text_words)


# load a model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

lables = [
    "Negative",
    'Neutral',
    'Positive'
]

# sentiment analysis
encoded_text = tokenizer(text_proc, return_tensors='pt')
output = model(**encoded_text)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    l = lables[i]
    s = scores[i]
    print (l, s)