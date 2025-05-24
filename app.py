import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse, parse_qs
import os 

app = Flask(__name__)

# --- Inisialisasi dan Persiapan Model ---
DATA_PATH = os.path.join('data', 'Dataset (2).xlsx') 
df = pd.read_excel(DATA_PATH, sheet_name='Final-BioSMA')

df["input_text"] = (
    df["Mata Pelajaran"] + " " +
    df["Materi"] + " " +
    df["Sub Materi"].fillna("") + " " +
    df["Jenjang"].astype(str) 
)
df["output_text"] = df['Jenjang'].astype(str) + " | " + df["Materi"] + " | " + df["Sub Materi"] + " | " + df["Link"]

# Inisialisasi dan Adaptasi TextVectorization
vectorizer = TextVectorization(output_mode="tf-idf", max_tokens=10000, ngrams=(1,2))
text_ds = tf.data.Dataset.from_tensor_slices(df["input_text"]).batch(32)
vectorizer.adapt(text_ds)

X_input = vectorizer(df["input_text"])

# Autoencoder Model
input_dim = X_input.shape[1]
input_layer = tf.keras.Input(shape=(input_dim,))

# Encoder
x = tf.keras.layers.Dense(512, activation="relu")(input_layer)
x = tf.keras.layers.Dense(256, activation="relu")(x)
encoded = tf.keras.layers.Dense(64, activation="relu")(x) 

# Decoder
x = tf.keras.layers.Dense(256, activation="relu")(encoded)
x = tf.keras.layers.Dense(512, activation="relu")(x)
decoded = tf.keras.layers.Dense(input_dim, activation="linear")(x)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.fit(X_input,
                X_input, 
                epochs=100, 
                verbose=0)

encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
embedded_vectors = encoder.predict(X_input)

# Fungsi Rekomendasi
def get_recommendations(query, df, vectorizer, encoder, embedded_vectors, top_n=5):
    query_vectorized = vectorizer([query])
    query_embedded = encoder.predict(query_vectorized)
    similarities = cosine_similarity(query_embedded, embedded_vectors).flatten()
    most_similar_indices = similarities.argsort()[-top_n:][::-1]
    return df.loc[most_similar_indices, ["Jenjang","Materi", "Sub Materi", "Link", "output_text"]]

# --- Rute Flask ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_query = request.form['query']
    recommendations = get_recommendations(user_query, df, vectorizer, encoder, embedded_vectors)
    recommendations_list = recommendations.to_dict(orient='records')
    return render_template('index.html', query=user_query, recommendations=recommendations_list)

def youtube_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    return None

app.jinja_env.filters['youtube_id'] = youtube_id

if __name__ == '__main__':
    app.run(debug=True)