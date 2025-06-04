import pandas as pd
import re

def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

def load_movie_features(csv_path='./ml-32m/movies.csv'):
    movies = pd.read_csv(csv_path)
    movies['year'] = movies['title'].apply(extract_year)
    movies = movies.dropna(subset=['year'])
    movies['year'] = movies['year'].astype(int)

    genres_split = movies['genres'].str.get_dummies(sep='|')

    year_min = movies['year'].min()
    year_max = movies['year'].max()
    year_norm = (movies['year'] - year_min) / (year_max - year_min)
    year_bucket = (year_norm * 20).round()

    movies['year_feat'] = 'year_' + year_bucket.astype(int).astype(str)

    movie_features = pd.concat([movies[['movieId', 'year_feat']], genres_split], axis=1)

    def extract_features(row):
        genres = [col for col in genres_split.columns if row[col] == 1]
        return genres + [row['year_feat']]

    movie_features['feature_list'] = movie_features.apply(extract_features, axis=1)

    movie_features = movie_features.set_index('movieId')[['feature_list']]

    return movie_features




