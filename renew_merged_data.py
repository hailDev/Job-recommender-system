import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from sklearn.model_selection import train_test_split

data_project = pd.read_csv('generate_random_data_project.csv',names=["id_project", "game_type"])
freelancer_preference = pd.read_csv('new_combined_preferences.csv')

result_data = []

# Loop melalui setiap baris di freelancer_preference
for _, row in freelancer_preference.iterrows():
    id_freelancer = row['id_freelancer']
    
    # Loop melalui setiap kolom kecuali 'id_freelancer'
    for col in freelancer_preference.columns[1:]:
        game_type = col
        preference_freelancer = row[col]
        
        # Filter data_project berdasarkan game_type
        filtered_data_project = data_project[data_project['game_type'] == game_type]
        
        # Loop melalui setiap baris di data_project yang sesuai
        for _, project_row in filtered_data_project.iterrows():
            id_project = project_row['id_project']
            
            # Tambahkan data ke list
            result_data.append({'id_project': id_project, 'id_freelancer': id_freelancer, 'game_type': game_type, 'preference_freelancer': preference_freelancer})

result = pd.DataFrame(result_data)
result.to_csv('new_merged_data.csv', index=False)