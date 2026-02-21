
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str, remove_stop=True, use_lemma=True, lowercase=True, alpha_only=True, strip_punctuation=True):
    doc = nlp(text)

    cleaned = []
    for token in doc:
        # Apply filters
        if strip_punctuation and token.is_punct:
            continue
        if remove_stop and token.is_stop:
            continue
        if alpha_only and not token.is_alpha:
            continue

        # Choose lemma or original text
        word = token.lemma_ if use_lemma else token.text

        # Apply lowercase
        if lowercase:
            word = word.lower()

        cleaned.append(word)
    return cleaned

def recap(text: str, remove_stop=True, use_lemma=True, lowercase=True, alpha_only=True, strip_punctuation=True):
    print(f"Original text: << {text} >>")
    print(f"Preprocessed tokens: << {preprocess_text(text, remove_stop, use_lemma, lowercase, alpha_only, strip_punctuation)} >>")
    print()

file = open("data.txt", "r")
sentences = file.read().split('\n')


i = 0
for l in sentences:
    if i == 7 or i == 8:
        recap(l, lowercase=False, alpha_only=False)
    else:
        recap(l, lowercase=False)
    i += 1

