import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from sklearn.model_selection import train_test_split


projects = pd.read_csv('generate_random_data_project.csv')
preferences = pd.read_csv('new_merged_data.csv')

n_project = preferences.id_project.nunique()
n_freelancer = preferences.id_freelancer.nunique()

train, test = train_test_split(preferences, test_size=0.2)

EMBEDDING_DIM = 30

#input layer
project_input = Input(shape=1)
freelancer_input = Input(shape=1)

#embedding layer
project_embedding = Embedding(n_project+1, EMBEDDING_DIM)(project_input)
freelancer_embedding = Embedding(n_freelancer+1, EMBEDDING_DIM)(freelancer_input)

#flatten layer
project_flat = Flatten()(project_embedding)
freelancer_flat = Flatten()(freelancer_embedding)

#output layer
output = Dot(1)([project_flat, freelancer_flat])

model = Model([project_input, freelancer_input],[output])

model.compile(optimizer='adam', loss='mse')

model.fit(x=[train.id_project, train.id_freelancer], y=[train.preference_freelancer],
         epochs=30, batch_size=128)

model.evaluate(x=[test.id_project, test.id_freelancer], y=[test.preference_freelancer])

model.save('recommender_model_tes.h5')