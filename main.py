import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from sklearn.model_selection import train_test_split

app = Flask(__name__)

model = load_model('recommender_model.h5')
projects = pd.read_csv('generate_random_data_project.csv')


def get_recommendations(id_freelancer, projects, model):
    try:
        projects = projects.copy()
        id_freelancers = np.array([id_freelancer] * len(projects))
        results = model([projects.id_project.values, id_freelancers]).numpy().reshape(-1)

        projects['predicted_rating'] = pd.Series(results)
        projects = projects.sort_values('predicted_rating', ascending=False)

        print(f'Recommendations for user {id_freelancer}')
        return projects
    except tf.errors.InvalidArgumentError:
        print(f'User {id_freelancer} not found. Returning random recommendations.')
        # Menggunakan np.random.randint untuk mendapatkan nilai random_state yang berbeda setiap kali dijalankan
        random_state = np.random.randint(1, 100)
        projects = projects.sample(n=10, random_state=random_state)
        return projects


@app.route('/get_recommendations', methods=['GET'])
def recommendations():
    rekomendasi = get_recommendations(34, projects, model)
    rekomendasi_dict = rekomendasi.to_dict(orient='records')
    return jsonify(rekomendasi_dict)

if __name__ == "__main__":
    app.run(debug=True)