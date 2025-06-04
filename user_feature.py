
import numpy as np

def build_user_feature_list(liked_genres, preferred_year_range):
    year_min, year_max = 1874, 2023
    year_norm = (np.mean(preferred_year_range) - year_min) / (year_max - year_min)
    year_bucket = round(year_norm * 20)
    year_feat = f'year_{year_bucket}'

    features = []
    for genre in liked_genres:
        features.append(genre)
    features.append(year_feat)
    return features


