import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from sklearn.model_selection import train_test_split

model = tf.keras.load_model('recommender_model.h5')
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

# Contoh pemanggilan fungsi
get_recommendations(19, projects, model)
