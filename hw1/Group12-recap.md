
# Group 12 Wokshop Assignment #1

## Time Spent

1 hour 50 minutes

## Participants

- Vasiliy Sukharev (sugharevmail@gmail.com)
- Arsenij Ivashenko	(dzonforevord@gmail.com)
- Kyrylo Filchenkov	(kirilfilch@gmail.com)
- Yaroslav Prokopishin	(yarik_prokopishin@knu.ua)
- Oluwadamilola Osundolire (Osundolire.d@gmail.com)
- Dahlia Sarah Salemi	(dahlia_sarah.salemi@g.enp.edu.dz)

## Original texts

```
"I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most."

Please sell these in Mexico!! "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!

one of the oldest confectionery Firms in the United States, now a Subsidiary of the Hershey Company, the Company was established in 1845 as Young and Smylie, they also make Apple Licorice Twists, Green Color and Blue Raspberry Licorice Twists, I like them all !

tweets : PROOF THAT YOUR VOICE ACTUALLY MATTERS! 🙏🏼🙌🏼💪🏼😭

YES LA KEEP LEADING THE WAY ❤️ #LoveForever

"Artificial intelligence could help detect breast cancer earlier, study finds."

The battery life is amazing, but the camera quality is disappointing in low light.

1 Social 2 media platforms are using artificial intelligence to detect harmful content!

"Omg deep cut, agreed https://t.co/0yGlkuo3bJ"
```

## Program Itself

```python
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


```

## Program Output

```command
$ python3 main.py
Original text: << "I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most." >>
Preprocessed tokens: << ['buy', 'Vitality', 'can', 'dog', 'food', 'product', 'find', 'good', 'quality', 'product', 'look', 'like', 'stew', 'process', 'meat', 'smell', 'well', 'Labrador', 'finicky', 'appreciate', 'product', 'well'] >>

Original text: << Please sell these in Mexico!! "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!! >>
Preprocessed tokens: << ['sell', 'Mexico', 'live', 'yr', 'miss', 'Twizzlers'] >>

Original text: << one of the oldest confectionery Firms in the United States, now a Subsidiary of the Hershey Company, the Company was established in 1845 as Young and Smylie, they also make Apple Licorice Twists, Green Color and Blue Raspberry Licorice Twists, I like them all ! >>
Preprocessed tokens: << ['old', 'confectionery', 'Firms', 'United', 'States', 'Subsidiary', 'Hershey', 'Company', 'Company', 'establish', 'Young', 'Smylie', 'Apple', 'Licorice', 'Twists', 'Green', 'Color', 'Blue', 'Raspberry', 'Licorice', 'Twists', 'like'] >>

Original text: << tweets : PROOF THAT YOUR VOICE ACTUALLY MATTERS! 🙏🏼🙌🏼💪🏼😭 >>
Preprocessed tokens: << ['tweet', 'proof', 'voice', 'ACTUALLY', 'matter'] >>

Original text: << YES LA KEEP LEADING THE WAY ❤️ #LoveForever >>
Preprocessed tokens: << ['yes', 'LA', 'lead', 'way', 'LoveForever'] >>

Original text: << "Artificial intelligence could help detect breast cancer earlier, study finds." >>
Preprocessed tokens: << ['artificial', 'intelligence', 'help', 'detect', 'breast', 'cancer', 'early', 'study', 'find'] >>

Original text: << The battery life is amazing, but the camera quality is disappointing in low light. >>
Preprocessed tokens: << ['battery', 'life', 'amazing', 'camera', 'quality', 'disappointing', 'low', 'light'] >>

Original text: << 1 Social 2 media platforms are using artificial intelligence to detect harmful content! >>
Preprocessed tokens: << ['1', 'Social', '2', 'medium', 'platform', 'artificial', 'intelligence', 'detect', 'harmful', 'content'] >>

Original text: << "Omg deep cut, agreed https://t.co/0yGlkuo3bJ" >>
Preprocessed tokens: << ['omg', 'deep', 'cut', 'agree', 'https://t.co/0yGlkuo3bJ'] >>

Original text: <<  >>
Preprocessed tokens: << [] >>
```

## Reflection

### What challenges did you encounter?

One of the curious things we've encountered was the way spaCy parses the sequence of a specific punctuation characters. For instance, given the sentence
```
Please sell these in Mexico!!"I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!
```
the preprocessed tokens are:
```
['sell', 'live', 'yr', 'miss', 'Twizzlers']
```
Here the word Mexico which, obviously, plays a signifficant role in the sentence, is stripped. Conersely, the sentence
```
Please sell these in Mexico!! "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!
```
gives us the proper sequence:
```
['sell', 'Mexico', 'live', 'yr', 'miss', 'Twizzlers']
```

One more thing were URLs, which are not considered as `alpha`, so should be passed with `alpha_only=False`:
```
"Omg deep cut, agreed https://t.co/0yGlkuo3bJ"
```
Results to:
```
['omg', 'deep', 'cut', 'agree', 'https://t.co/0yGlkuo3bJ']
```

### Which preprocessing choices felt uncertain or difficult?

Because of the sentiment analysis it is better to keep stop words. The example is
```
The battery life is amazing, but the camera quality is disappointing in low light.
```
which preprocessed without 'but' in the middle with the `remove_stop=True`:
```
['battery', 'life', 'amazing', 'camera', 'quality', 'disappointing', 'low', 'light']
```

Another example is the sentence we've already encountered:
```
Please sell these in Mexico!! "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!
```

### What patterns did you notice in your cleaned text? 

Nouns are always kept since they can't be treated either as stop words, etc. Usually the same thing about verbs. Also, the original sentence become a shorter and far more convenient to reason about.



