import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

#use defined pre-processing function from other .py files
#nltk.download('punkt')
#nltk.download('stopwords')

# pre-trained BERT model for generating embeddings
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'doc_sum', 'article100.csv'))


preprocessed_data_path = "preprocessed_data.pkl"
embeddings_path = "article_embeddings.npy"

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    preprocessed_text = " ".join(filtered_tokens)
    
    return preprocessed_text

if os.path.exists(preprocessed_data_path):
    with open(preprocessed_data_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
else:
    data['preprocessed_content'] = data['content'].apply(preprocess)
    
    with open(preprocessed_data_path, 'wb') as f:
        pickle.dump(data['preprocessed_content'], f)

if os.path.exists(embeddings_path):
    article_embeddings = np.load(embeddings_path)
else:
    article_embeddings = model.encode(data['preprocessed_content'].tolist(), show_progress_bar=True)
    
    np.save(embeddings_path, article_embeddings)

def expand_query(query, num_expansions=10):
    preprocessed_query = preprocess(query)
    
    # Query embedding
    query_embedding = np.array(model.encode(preprocessed_query)).reshape(1, -1)  # Convert to numpy array and reshape to make it 2D
    
    # cosine similarity b/w  query embedding and article embeddings
    similarities = cosine_similarity(query_embedding, article_embeddings)
    
    # top similar docs
    top_indices = similarities.argsort(axis=1).flatten()[-num_expansions:]
    
    expanded_query = data.iloc[top_indices]['title'].tolist()
    return expanded_query

#Test
user_query = "Trump and Taxes"
expanded_query = expand_query(user_query)
print("User Query:", user_query)
for item in expanded_query:
    print('Expanded Query:', item)
