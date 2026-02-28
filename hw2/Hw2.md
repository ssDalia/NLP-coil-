# Assignement 2: Text Vectorization with FastText

## Objective
Explore how to transform text into meaningful numerical representations using FastText embeddings, and investigate real-world class imbalance in text data.

---

1. **From Words to Numbers**: Different text representation methods
2. **FastText Embeddings**: How FastText creates dense vectors that capture meaning
3. **Real Dataset**: SMS Spam Collection with natural class imbalance
4. **Hands-On Vectorization**: Build a complete text-to-vector pipeline
5. **Visualization**: See how embeddings cluster by meaning

---
 __realized by  :__
  
```

Vasiliy Sukharev : sugharevmail@gmail.com
Arsenij Ivashenko : dzonforevord@gmail.com
Kyrylo Filchenkov : kirilfilch@gmail.com
Dahlia Sarah Salemi : dahlia_sarah.salemi@g.enp.edu.dz
```
time spent : approximately 2 hours

## Section 1: Setup and Installation

Install and import all required libraries.


```python
# Install required packages
!pip install fasttext pandas numpy matplotlib seaborn scikit-learn spacy tqdm
!python -m spacy download en_core_web_sm
```

    Collecting fasttext
      Downloading fasttext-0.9.3.tar.gz (73 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/73.4 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.4/73.4 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (3.10.0)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.12/dist-packages (0.13.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
    Requirement already satisfied: spacy in /usr/local/lib/python3.12/dist-packages (3.8.11)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (4.67.3)
    Collecting pybind11>=2.2 (from fasttext)
      Using cached pybind11-3.0.2-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: setuptools>=0.7.0 in /usr/local/lib/python3.12/dist-packages (from fasttext) (75.2.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.3)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (4.61.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (1.4.9)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (26.0)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib) (3.3.2)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /usr/local/lib/python3.12/dist-packages (from spacy) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.12/dist-packages (from spacy) (1.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.12/dist-packages (from spacy) (1.0.15)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.12/dist-packages (from spacy) (2.0.13)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.12/dist-packages (from spacy) (3.0.12)
    Requirement already satisfied: thinc<8.4.0,>=8.3.4 in /usr/local/lib/python3.12/dist-packages (from spacy) (8.3.10)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /usr/local/lib/python3.12/dist-packages (from spacy) (1.1.3)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.12/dist-packages (from spacy) (2.5.2)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.12/dist-packages (from spacy) (2.0.10)
    Requirement already satisfied: weasel<0.5.0,>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from spacy) (0.4.3)
    Requirement already satisfied: typer-slim<1.0.0,>=0.3.0 in /usr/local/lib/python3.12/dist-packages (from spacy) (0.24.0)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from spacy) (2.32.4)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in /usr/local/lib/python3.12/dist-packages (from spacy) (2.12.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from spacy) (3.1.6)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (2.41.4)
    Requirement already satisfied: typing-extensions>=4.14.1 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (4.15.0)
    Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.4.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2026.1.4)
    Requirement already satisfied: blis<1.4.0,>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from thinc<8.4.0,>=8.3.4->spacy) (1.3.3)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.12/dist-packages (from thinc<8.4.0,>=8.3.4->spacy) (0.1.5)
    Requirement already satisfied: typer>=0.24.0 in /usr/local/lib/python3.12/dist-packages (from typer-slim<1.0.0,>=0.3.0->spacy) (0.24.1)
    Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /usr/local/lib/python3.12/dist-packages (from weasel<0.5.0,>=0.4.2->spacy) (0.23.0)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /usr/local/lib/python3.12/dist-packages (from weasel<0.5.0,>=0.4.2->spacy) (7.5.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->spacy) (3.0.3)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.12/dist-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.4.2->spacy) (2.1.1)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (8.3.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer>=0.24.0->typer-slim<1.0.0,>=0.3.0->spacy) (0.1.2)
    Using cached pybind11-3.0.2-py3-none-any.whl (310 kB)
    Building wheels for collected packages: fasttext
      Building wheel for fasttext (pyproject.toml) ... [?25l[?25hdone
      Created wheel for fasttext: filename=fasttext-0.9.3-cp312-cp312-linux_x86_64.whl size=4647421 sha256=52232374901df049d279600090f09f6396ba1e2b6fd0e3de0b87760cd981de05
      Stored in directory: /root/.cache/pip/wheels/20/27/95/a7baf1b435f1cbde017cabdf1e9688526d2b0e929255a359c6
    Successfully built fasttext
    Installing collected packages: pybind11, fasttext
    Successfully installed fasttext-0.9.3 pybind11-3.0.2
    Collecting en-core-web-sm==3.8.0
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m94.8 MB/s[0m eta [36m0:00:00[0m
    [?25h[38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_sm')
    [38;5;3m⚠ Restart to reload dependencies[0m
    If you are in a Jupyter or Colab notebook, you may need to restart Python in
    order to load all the package's dependencies. You can do this by selecting the
    'Restart kernel' or 'Restart runtime' option.
    


```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
from numpy.linalg import norm
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print(" libraries imported successfully!")
```

     libraries imported successfully!
    

## Section 2: Understanding Text Representation Methods


# __1.  Bag-of-Words (BoW)__

###__Concept:__<br>
Bag-of-Words (BoW) represents each text as a vector of word counts, ignoring word order. It is simple, fast, and often effective for basic text classification tasks.

__Limitation:__<br>
BoW cannot distinguish sentences with the same words but different meaning. For example:


```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love NLP",
    "I love Python",
    "The dog bit the man",
    "The man bit the dog"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow_matrix.toarray())
```

    Vocabulary: ['bit' 'dog' 'love' 'man' 'nlp' 'python' 'the']
    BoW Matrix:
     [[0 0 1 0 1 0 0]
     [0 0 1 0 0 1 0]
     [1 1 0 1 0 0 2]
     [1 1 0 1 0 0 2]]
    

###__Experimentation__: Using bigrams to capture word order:


```python
vectorizer_bigram = CountVectorizer(ngram_range=(1,2))
bow_matrix_bigram = vectorizer_bigram.fit_transform(sentences)

print("Vocabulary with bigrams:", vectorizer_bigram.get_feature_names_out())
print("BoW matrix with bigrams:\n", bow_matrix_bigram.toarray())
```

    Vocabulary with bigrams: ['bit' 'bit the' 'dog' 'dog bit' 'love' 'love nlp' 'love python' 'man'
     'man bit' 'nlp' 'python' 'the' 'the dog' 'the man']
    BoW matrix with bigrams:
     [[0 0 0 0 1 1 0 0 0 1 0 0 0 0]
     [0 0 0 0 1 0 1 0 0 0 1 0 0 0]
     [1 1 1 1 0 0 0 1 0 0 0 2 1 1]
     [1 1 1 0 0 0 0 1 1 0 0 2 1 1]]
    

####__Observation:__

* Including bigrams like __"the dog"__ and __"the man"__ allows the model to partially capture word order.

* Now __"The dog bit the man"__ and __"The man bit the dog"__ have different vectors thus reducing ambiguity.

##__Visualizing the BoW matrix:__


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(bow_matrix_bigram.toarray(), columns=vectorizer_bigram.get_feature_names_out())
sns.heatmap(df, annot=True, cmap="YlGnBu")
plt.title("BoW Matrix Heatmap with Bigrams")
plt.show()
```


    
![png](Hw2_files/Hw2_11_0.png)
    


* The heatmap visually shows which words or bigrams appear in each sentence.

* It makes the representation easy to interpret and highlights ambiguities.

### __Conclusion:__

Bow is an intuitive and effective representation for short texts or small datasets. However it ignores semantics and word order. Using n-grams can mitigate some limitations by capturing local word sequences. Visualizations such as heatmaps help understand how the texts are represented and where ambiguities may exist.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Concept**:

TF-IDF improves upon Bag-of-Words by weighting words according to their importance in a document relative to the entire corpus.

TF (Term Frequency) =  how often a word appears in a document

IDF (Inverse Document Frequency) = how rare the word is across all documents

Rare words receive higher weight while common words receive lower weight.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
# uncomment next line to remove stop word "the"
# vectorizer = TfidfVectorizer(stop_words='english')

sentences = [
    "the nlp",
    "the python",
    "the dog bit the man",
    "the man bit the dog"
]

tfidf_matrix = vectorizer.fit_transform(sentences)
words = vectorizer.get_feature_names_out()

df = pd.DataFrame(tfidf_matrix.toarray(), columns=words)
print("TF-IDF Matrix:")
print(df.round(2))

print("\nIndividual Word Rarity (IDF Scores):")
idf_values = dict(zip(words, vectorizer.idf_))
for word, score in idf_values.items():
    print(f"Word: {word:10} | IDF Weight: {score:.4f}")
```

    TF-IDF Matrix:
        bit   dog   man   nlp  python   the
    0  0.00  0.00  0.00  0.89    0.00  0.46
    1  0.00  0.00  0.00  0.00    0.89  0.46
    2  0.46  0.46  0.46  0.00    0.00  0.61
    3  0.46  0.46  0.46  0.00    0.00  0.61
    
    Individual Word Rarity (IDF Scores):
    Word: bit        | IDF Weight: 1.5108
    Word: dog        | IDF Weight: 1.5108
    Word: man        | IDF Weight: 1.5108
    Word: nlp        | IDF Weight: 1.9163
    Word: python     | IDF Weight: 1.9163
    Word: the        | IDF Weight: 1.0000
    

### __Observations:__

* Common words receive lower weights
The word "the" appears in every sentence, so its IDF score is low.
This means TF-IDF automatically reduces its importance.

* Rare words receive higher weights
Words like "nlp" and "python" appear in only one sentence.
They receive higher IDF scores and therefore higher TF-IDF values.

---

__Conclusion (TF-IDF):__

TF-IDF improves over Bag-of-Words by reducing the influence of frequent words and emphasizing rare informative ones. However it still does not capture semantic meaning or word order. It remains a frequency based statistical representation rather than a contextual one.

---
## Section 3: FastText Embeddings


### Download Pre-trained FastText Model


```python
# Download pre-trained FastText model for English
# Trained on Common Crawl (billions of words, ~2M vocabulary)
# Note: This is ~4.5GB and may take some time

import fasttext.util

print("📥 Downloading FastText model... (this may take a some time)")
fasttext.util.download_model('en', if_exists='ignore')
ft_model = fasttext.load_model('cc.en.300.bin')

print("\n✓ FastText model loaded!")
print(f"✓ Vocabulary size: {len(ft_model.words):,} words")
print(f"✓ Vector dimension: {ft_model.get_dimension()}")
```

    📥 Downloading FastText model... (this may take a some time)
    Downloading https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    
    
    ✓ FastText model loaded!
    ✓ Vocabulary size: 2,000,000 words
    ✓ Vector dimension: 300
    

### Exploration Questions :
__1. Semantic Similarity test__ : trying pairs  


```python
# Function to calculate similarity between two words
def get_similarity(word1, word2):
    """Calculate cosine similarity between two words"""
    vec1 = ft_model.get_word_vector(word1)
    vec2 = ft_model.get_word_vector(word2)
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity

# Test semantic similarity
print(f"'car' vs 'automobile': {get_similarity('car', 'automobile'):.4f}")
print(f"'king' vs 'queen': {get_similarity('king', 'queen'):.4f}")
print(f"'computer' vs 'keyboard': {get_similarity('computer', 'keyboard'):.4f}")
print(f"'love' vs 'war': {get_similarity('love', 'war'):.4f}")
print(f"'banana' vs 'democracy': {get_similarity('banana', 'democracy'):.4f}")
print(f"'play' vs 'playing': {get_similarity('play', 'playing'):.4f}")
print(f"'happy' vs 'happy': {get_similarity('happy', 'happy'):.4f}") #here we use the same word to check that the cosine similarity is 1


print("\n🎯 High similarity (>0.8) = semantically close")
print("🎯 Low similarity (<0.3) = opposite or unrelated")
print("🎯 FastText understands morphology: 'play' and 'playing' are very similar!")
```

    'car' vs 'automobile': 0.7022
    'king' vs 'queen': 0.7069
    'computer' vs 'keyboard': 0.4811
    'love' vs 'war': 0.1400
    'banana' vs 'democracy': 0.2242
    'play' vs 'playing': 0.7612
    'happy' vs 'happy': 1.0000
    
    🎯 High similarity (>0.8) = semantically close
    🎯 Low similarity (<0.3) = opposite or unrelated
    🎯 FastText understands morphology: 'play' and 'playing' are very similar!
    

###__Semantic similarity examples:__

__'play' vs 'playing'__: 0.7612 <br>
__'car' vs 'automobile'__: 0.7022 <br>
__'king' vs 'queen'__: 0.7069 <br>
__'computer' vs 'keyboard'__: 0.4811 <br>
__'love' vs 'war'__: 0.1400 <br>
__'banana' vs 'democracy'__: 0.2242 <br>

---

__Most similar pair:__<br>
__'play' vs 'playing'__: __0.7612__<br>
FastText breaks words into subword units using n-grams. When one word is a subword of the other FastText captures this morphological relationship which results in a high similarity score.

---

__Least similar pair:__<br>
'love' vs 'war': __0.1400__<br>
they re complete opposites or at least contradictory so the similarity score is __< 0.2__

---

After testing additional word pairs, we found that "car" and "automobile" had one of the highest similarity scores with __0.7022__, confirming that FastText captures synonymy effectively. In contrast, unrelated words such as "banana" and "democracy" showed very low similarity with a similarity score of __0.2242__, demonstrating that the embedding space reflects semantic distance. Morphologically related words like "play" and "playing" also showed high similarity confirming FastText’s ability to use subword information.


### Testing with Typos and Unknown Words


```python
# Test with typos and unknown words
print("Handling unknown/misspelled words:\n")

print(f"'amazingg' (typo) vs 'amazing': {get_similarity('amazingg', 'amazing'):.4f}")
print(f"'happyyy' (typo) vs 'happy': {get_similarity('happyyy', 'happy'):.4f}")
print(f"'suuuuper' (typo) vs 'super': {get_similarity('suuuuper', 'super'):.4f}")
```

    Handling unknown/misspelled words:
    
    'amazingg' (typo) vs 'amazing': 0.5117
    'happyyy' (typo) vs 'happy': 0.4121
    'suuuuper' (typo) vs 'super': 0.7021
    

###__Observation:__

Despite spelling errors, similarity scores remain relatively high.

__Explanation:__

FastText uses subword units (n-grams) to build word representations.
Because misspelled words still share many character fragments with the correct word, FastText can recognize their similarity.

__For example:__

* "happyyy" shares subword fragments with "happy"

* "amazingg" shares fragments with "amazing"

__This makes FastText robust to:__

* Typos

* Informal writing

* Unknown words

---

## Section 4: Load Real Dataset - SMS Spam Collection

We'll use the **SMS Spam Collection** dataset:
- **Size**: ~5,574 SMS messages
- **Classes**: ham (legitimate) and spam
- **Source**: UCI Machine Learning Repository
- **Naturally imbalanced**: Real-world distribution!


```python
# Download SMS Spam Collection dataset
import urllib.request
import zipfile
import os

# Download dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
zip_path = 'smsspamcollection.zip'

if not os.path.exists('SMSSpamCollection'):
    print("📥 Downloading SMS Spam Collection dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    os.remove(zip_path)
    print("✓ Download complete!")
else:
    print("✓ Dataset already exists!")

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

print(f"\n✓ Dataset loaded: {len(df):,} messages")
print(f"\nFirst few rows:")
print(df.head())
```

    📥 Downloading SMS Spam Collection dataset...
    📦 Extracting...
    ✓ Download complete!
    
    ✓ Dataset loaded: 5,572 messages
    
    First few rows:
      label                                               text
    0   ham  Go until jurong point, crazy.. Available only ...
    1   ham                      Ok lar... Joking wif u oni...
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3   ham  U dun say so early hor... U c already then say...
    4   ham  Nah I don't think he goes to usf, he lives aro...
    

### Calculate Baseline Accuracy

If we always predict "ham" (the majority class), what accuracy would we get?


```python
# Calculate baseline accuracy (majority class proportion)
baseline_accuracy = df['label'].value_counts(normalize=True).max()

print(f"Baseline Accuracy (always predict 'ham'): {baseline_accuracy:.1%}")

print("\n⚠️ Any model we build MUST beat this baseline!")
print("If our model gets 87% accuracy, that's barely better than always guessing 'ham'.")
```

    Baseline Accuracy (always predict 'ham'): 86.6%
    
    ⚠️ Any model we build MUST beat this baseline!
    If our model gets 87% accuracy, that's barely better than always guessing 'ham'.
    

##Exploration question :

__3.Baseline accuracy:__ Why is it important to calculate this BEFORE building models? <br>

It is important to compute baseline accuracy with a simple model because it allows us to detect early class imbalance. For example, if we have two classes where one is majority (70%) and the other minority (30%), a baseline model that always predicts the majority class would achieve 70% accuracy. In other words, it answers the question: “If I do nothing complicated, what accuracy can I get?”. This gives us hints about class imbalance and helps us decide how to treat the problem to improve model performance.


---

## Section 5: Build Text Vectorization Pipeline

Now let's convert our SMS messages into FastText vectors!

### Text Preprocessing with spaCy


```python
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def vectorize_text(text, ft_model):
    """
    Convert text to FastText vector.

    Steps:
    1. Clean text with spaCy (lowercase, remove stopwords/punctuation, lemmatize)
    2. Get FastText vector for each cleaned token
    3. Average vectors to create document vector

    Args:
        text: Input text string
        ft_model: Loaded FastText model

    Returns:
        numpy array of shape (300,) representing the document
    """
    # Process text with spaCy
    doc = nlp(text.lower())

    # Extract clean tokens (lemmatized, no stopwords, no punctuation)
    clean_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]

    # Edge case: if no valid tokens, return zero vector
    if len(clean_tokens) == 0:
        return np.zeros(ft_model.get_dimension())

    # Get FastText vector for each token
    vectors = [ft_model.get_word_vector(token) for token in clean_tokens]

    # Average all vectors to create document vector
    doc_vector = np.mean(vectors, axis=0)

    return doc_vector

# Test on example
test_text = "FREE! Win a prize now! Call 123-456-7890"
test_vector = vectorize_text(test_text, ft_model)

print(f"Input: '{test_text}'")
print(f"\nOutput vector shape: {test_vector.shape}")
print(f"First 10 dimensions: {test_vector[:10]}")
```

    Input: 'FREE! Win a prize now! Call 123-456-7890'
    
    Output vector shape: (300,)
    First 10 dimensions: [-0.07042163 -0.08417823  0.08138885 -0.05110617  0.10062113 -0.04342021
     -0.09545817 -0.04873976 -0.07071517 -0.02255991]
    

##Exploration question :
__4.Text cleaning:__ we remove the stopword filtering


```python
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def vectorize_stopwords_text(text, ft_model):

    # Process text with spaCy
    doc = nlp(text.lower())

    # Extract tokens without stopwords filtering (lemmatized, we keep stopwords, no punctuation)
    tokens_without_stopwords_filtering = [
        token.lemma_ for token in doc
        if  not token.is_punct and token.lemma_.strip()
    ]

    # Edge case: if no valid tokens, return zero vector
    if len(tokens_without_stopwords_filtering) == 0:
        return np.zeros(ft_model.get_dimension())

    # Get FastText vector for each token
    vectors = [ft_model.get_word_vector(token) for token in tokens_without_stopwords_filtering]

    # Average all vectors to create document vector
    doc_vector = np.mean(vectors, axis=0)

    return doc_vector

# Test on example
test_text = "FREE! Win a prize now! Call 123-456-7890"
test_vector = vectorize_stopwords_text(test_text, ft_model)

print(f"Input: '{test_text}'")
print(f"\nOutput vector shape: {test_vector.shape}")
print(f"First 10 dimensions: {test_vector[:10]}")
```

    Input: 'FREE! Win a prize now! Call 123-456-7890'
    
    Output vector shape: (300,)
    First 10 dimensions: [-0.04183479 -0.09134568  0.06105985 -0.02326233  0.03359714 -0.03885876
     -0.02959773 -0.05069644 -0.0452666  -0.036339  ]
    

### Show Text Cleaning Effect


```python
# Show what happens during cleaning
doc = nlp(test_text.lower())
original_tokens = [token.text for token in doc]
clean_tokens = [
    token.lemma_ for token in doc
    if not token.is_stop and not token.is_punct and token.lemma_.strip()
]

print(f"Original tokens: {original_tokens}")
print(f"After cleaning: {clean_tokens}")
print(f"\n📝 From {len(original_tokens)} tokens → {len(clean_tokens)} meaningful words")

# Show what happens during if we keep stopwords
doc = nlp(test_text.lower())
original_tokens = [token.text for token in doc]
tokens_without_stopwords_filtering = [
    token.lemma_ for token in doc
    if  not token.is_punct and token.lemma_.strip()
]

print(f"Original tokens: {original_tokens}")
print(f"After preprocessing without removing stopwords: {tokens_without_stopwords_filtering }")
print(f"\n📝 From {len(original_tokens)} tokens → {len(tokens_without_stopwords_filtering )} meaningful words + stopwords")
```

    Original tokens: ['free', '!', 'win', 'a', 'prize', 'now', '!', 'call', '123', '-', '456', '-', '7890']
    After cleaning: ['free', 'win', 'prize', '123', '456', '7890']
    
    📝 From 13 tokens → 6 meaningful words
    Original tokens: ['free', '!', 'win', 'a', 'prize', 'now', '!', 'call', '123', '-', '456', '-', '7890']
    After preprocessing without removing stopwords: ['free', 'win', 'a', 'prize', 'now', 'call', '123', '456', '7890']
    
    📝 From 13 tokens → 9 meaningful words + stopwords
    

### Apply to Entire Dataset

This will take ~2-3 minutes.


```python

```


```python
# Enable progress bar
tqdm.pandas(desc="Vectorizing messages")


print("📊 Vectorizing messages WITHOUT stopwords...\n")
df['vector_no_stop'] = df['text'].progress_apply(
    lambda text: vectorize_text(text, ft_model)
)

print("\n📊 Vectorizing messages WITH stopwords...\n")
df['vector_with_stop'] = df['text'].progress_apply(
    lambda text: vectorize_stopwords_text(text, ft_model)
)

print("\n✓ Vectorization complete!")
```

    📊 Vectorizing messages WITHOUT stopwords...
    
    

    Vectorizing messages: 100%|██████████| 5572/5572 [01:00<00:00, 92.54it/s]
    

    
    📊 Vectorizing messages WITH stopwords...
    
    

    Vectorizing messages: 100%|██████████| 5572/5572 [01:01<00:00, 90.30it/s]

    
    ✓ Vectorization complete!
    

    
    

###__Comparing vectors (without stopwords filtering vs with):__


```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```


```python
print("\n Comparing vectors (with vs without stopwords)...\n")

df['similarity_between_versions'] = df.apply(
    lambda row: cosine_similarity(row['vector_no_stop'], row['vector_with_stop']),
    axis=1
)

print("Average cosine similarity between versions:",
      df['similarity_between_versions'].mean())

print("\nSample comparison:")
print("Text:", df.iloc[0]['text'])
print("Cosine similarity:", df.iloc[0]['similarity_between_versions'])
```

    
     Comparing vectors (with vs without stopwords)...
    
    Average cosine similarity between versions: 0.7876461841817509
    
    Sample comparison:
    Text: Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    Cosine similarity: 0.9488107562065125
    


__Text cleaning:__ so by removing the stopword filtering. How does this affect the vectors?


Removing stopword filtering affects the document vectors because additional word embeddings are included in the averaging process. When stopwords are kept their vectors contribute to the final representation, slightly shifting the document embedding.<br>

Although stopwords do not carry strong semantic meaning, they still have vector representations in FastText. Including them makes the document vector more neutral and slightly dilutes the influence of informative words.<br>

As a result, the numerical values of the embeddings change but  not drastically. The impact is more noticeable in short texts, where each word has a stronger influence on the average.<br>

in conclusion, removing stopwords produces cleaner and more content-focused document vectors.<br>

---

## Section 6: Visualize FastText Embeddings

Use dimensionality reduction to visualize our 300D vectors in 2D.

### Dimensionality Reduction: 300D → 2D


```python
# Convert vectors to numpy array
X = np.vstack(df['vector_no_stop'].values) #we use the vectors where we removed stop words
y = df['label'].values

print(f"Data shape: {X.shape}")
print("Reducing from 300D to 2D...\n")

# Step 1: PCA for initial reduction (faster)
print("Step 1: PCA from 300D to 50D")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# Step 2: t-SNE for visualization
print("Step 2: t-SNE from 50D to 2D")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X_pca)

print("\n✓ Dimensionality reduction complete!")
```

    Data shape: (5572, 300)
    Reducing from 300D to 2D...
    
    Step 1: PCA from 300D to 50D
    Step 2: t-SNE from 50D to 2D
    
    ✓ Dimensionality reduction complete!
    

### Create Scatter Plot

###__identifying overlapping zones between ham and spam :__


```python
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

plt.figure(figsize=(12, 8))

colors = {'ham': 'green', 'spam': 'red'}
for label in ['ham', 'spam']:
    mask = y == label
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=colors[label],
        label=label,
        alpha=0.5,
        s=50
    )

# density computing

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid_points = np.vstack([xx.ravel(), yy.ravel()])


data_ham = X_2d[y == 'ham'].T
data_spam = X_2d[y == 'spam'].T

# Kernel Density Estimation
kde_ham = gaussian_kde(data_ham)
kde_spam = gaussian_kde(data_spam)

z_ham = kde_ham(grid_points).reshape(xx.shape)
z_spam = kde_spam(grid_points).reshape(xx.shape)



# Normalising values
z_ham_norm = z_ham / z_ham.max()
z_spam_norm = z_spam / z_spam.max()


# if ham's or spam's density is 0 than overlap is by definition 0 (we take the min of both class densities )
z_overlap = np.minimum(z_ham_norm, z_spam_norm)


#we draw a red contour if indicator of overlap is > to 5% (i.e  both class densities are at least 5% )
contour = plt.contour(xx, yy, z_overlap, levels=[0.05], colors='red', linewidths=3)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=3)]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles + custom_lines, labels + ['Overlap Zone'], fontsize=12)

plt.title('overlap', fontsize=16, fontweight='bold')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](Hw2_files/Hw2_49_0.png)
    


##Exploration question:

###__2.Examining visualisation:__

In the visualization we can clearly observe regions where spam and ham messages overlap forming areas close to the decision boundary. These ambiguous messages often contain short, generic text that lacks strong indicators of either category. <br>For example, messages like:

```

Label: ham – "I know you are. Can you pls open the back?"

Label: ham – "See you at 5. Don't forget the keys."

Label: spam – "Congratulations! Claim your free gift now." (but short versions of promotions may resemble ham)

```

Such messages are ambiguous because:

Dimensionality reduction __PCA + t-SNE__ compressed the original 300-dimensional embeddings into 2D. While the main structure is preserved, some subtle distinctions between spam and ham are lost.

Short messages or messages that combine features of both categories like a short request with promotional content are harder to classify.

The lack of sufficient context  makes it difficult for the model to clearly separate them.

---

 __Pattern observation:__<br>
Ambiguous messages tend to be short, neutral, or combining characteristics of both classes, and they often lie in the overlapping regions of the embedding space after dimensionality reduction. This explains why some ham messages can appear in the spam cluster and vice versa,especially with the class inbalance making it more likely to predicted as a ham most of the time



---

## Section 7: Explore Class Imbalance Trade-offs

What if we balance the dataset? Let's see what we gain and lose.


```python
# Create balanced subset
min_class_size = df['label'].value_counts().min()

print(f"Smallest class (spam) has {min_class_size} messages")
print(f"\nCreating balanced subset...")

# Sample equally from each class
balanced_df = pd.concat([
    df[df['label'] == 'ham'].sample(min_class_size, random_state=42),
    df[df['label'] == 'spam'].sample(min_class_size, random_state=42)
])

print(f"\nOriginal dataset: {len(df):,} messages")
print(f"Balanced dataset: {len(balanced_df):,} messages")

print("\nBalanced distribution:")
print(balanced_df['label'].value_counts())
print(balanced_df['label'].value_counts(normalize=True))
```

    Smallest class (spam) has 747 messages
    
    Creating balanced subset...
    
    Original dataset: 5,572 messages
    Balanced dataset: 1,494 messages
    
    Balanced distribution:
    label
    ham     747
    spam    747
    Name: count, dtype: int64
    label
    ham     0.5
    spam    0.5
    Name: proportion, dtype: float64
    


```python
# Visualize original vs balanced
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original
df['label'].value_counts().plot(
    kind='bar',
    color=['green', 'red'],
    ax=axes[0]
)
axes[0].set_title('Original Dataset (Imbalanced)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Balanced
balanced_df['label'].value_counts().plot(
    kind='bar',
    color=['green', 'red'],
    ax=axes[1]
)
axes[1].set_title('Balanced Dataset', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()
```


    
![png](Hw2_files/Hw2_53_0.png)
    


## Exploration question :

###__5.Class imbalance__:  when is it appropriate ?

Class imbalance can reflect the true distribution of the data and is sometimes appropriate. For example, in fraud detection, most transactions are legitimate and only a few are fraudulent. Similarly, for rare diseases, the majority of patients are healthy. In these cases, forcing balanced classes could create unrealistic scenarios and reduce the model’s relevance. It is better to use evaluation metrics such as F1-score or AUC that account for imbalance rather than
relying only on accuracy.

