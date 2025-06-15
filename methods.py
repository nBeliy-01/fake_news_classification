import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

stopwods_set = set(stopwords.words("english"))
stopwods_set.add('reuters')

stemmer = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

# функція для видалення спецсимволів та токенізації (обробки тексту)
def process_text(sentence):
    regex_sub = re.compile(r'[^a-z0-9\s]+')
    cleaned_text = regex_sub.sub(' ', sentence.lower())
    word_list = [word.lower() for word in word_tokenize(cleaned_text) if word not in stopwods_set]
    return word_list

# функція для лематизації та стемінгу
def process_lemma_stemming(sentence):
    stemmed_list = [stemmer.stem(word) for word in sentence]
    lemmatized_list = [lemmatizer.lemmatize(word) for word in stemmed_list]
    return lemmatized_list

# функція для розрахунку основних метрик для BERT
def calculate_metrics(eval_pred):
    # Отримуємо передбачення та справжні мітки
    predictions, labels = eval_pred

    # softmax для отримання ймовірностей
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)

    # Беремо ймовірності позитивного класу для обчислення AUC
    positive_class_probs = probabilities[:, 1]

    # Обчислюємо AUC
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)

    # Отримуємо передбачені класи (найбільш імовірні)
    predicted_classes = np.argmax(predictions, axis=1)

    # Обчислюємо accuracy
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)

    return {"Accuracy": acc, "AUC": auc}

# функція для створення наборів даних для FastText
def process_fasttext(text, label, name=""):
    
    with open(f'fast-text/{name}.txt', 'w') as f:
        for elem, label in zip(text, label):
            if label == 0:
                line = f"__label__NOT_FAKE {elem}\n"
                f.write(line)
            else:
                line = f"__label__FAKE {elem}\n"
                f.write(line)