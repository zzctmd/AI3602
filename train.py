from lightfm import LightFM
from lightfm.data import Dataset
from scipy import sparse
import numpy as np

import pandas as pd
from tqdm import tqdm
import re
import joblib
import pickle
from lightfm.evaluation import auc_score
import random
from movie_feature import load_movie_features
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a LightFM model for movie recommendations.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--movie_csv', type=str, default='/Users/zzcnb123456/AI_project/recommond/ml-32m/movies.csv',
                        help='Path to the movies CSV file.')
    parser.add_argument('--ratings_csv', type=str, default='/Users/zzcnb123456/AI_project/recommond/ml-32m/ratings.csv')
    parser.add_argument('--output_model_path', type=str, default='model.pkl',
                        help='Path to save the trained model.')
    parser.add_argument('--output_dataset_path', type=str, default='dataset.pkl',
                        help='Path to save the dataset.')

    args = parser.parse_args()
    epochs = args.epochs
    movie_csv = args.movie_csv
    ratings_csv = args.ratings_csv
    output_model_path = args.output_model_path
    output_dataset_path = args.output_dataset_path


    # 加载电影特征和评分数据
    movie_features = load_movie_features(csv_path=movie_csv)
    ratings = pd.read_csv(ratings_csv)
    all_movie_features = set()
    for features in movie_features['feature_list']:
        all_movie_features.update(features)
    all_user_features = all_movie_features

    existing_user_ids = list(ratings['userId'].unique())

    dataset = Dataset()
    dataset.fit(users=existing_user_ids, items=ratings['movieId'].unique())
    dataset.fit_partial(
        items=movie_features.index.tolist(),
        item_features=all_movie_features,
        user_features=all_user_features
    )

    item_features = dataset.build_item_features(
        ((movie_id, features) for movie_id, features in movie_features['feature_list'].items())
    )

    user_feature_tuples = [(user_id, []) for user_id in existing_user_ids]  # 训练时用户特征全空
    user_features_matrix = dataset.build_user_features(user_feature_tuples)

    interactions = list(zip(ratings['userId'], ratings['movieId'], ratings['rating']))
    interactions_matrix, weights_matrix = dataset.build_interactions(interactions)

    model = LightFM(loss='warp',
                    no_components=50,
                    learning_rate=0.05,
                    item_alpha=1e-6,
                    user_alpha=1e-6)

    for epoch in tqdm(range(epochs)):
        model.fit_partial(interactions_matrix, epochs=1, num_threads=1,
                        user_features=user_features_matrix, item_features=item_features)
        subset_users = random.sample(list(dataset.mapping()[0].values()), k=1000)  # 采样1000个用户
        sub_interactions = interactions_matrix.tocsr()[subset_users, :]
        train_auc = auc_score(model, sub_interactions, user_features=user_features_matrix, item_features=item_features).mean()
        print(f'Epoch {epoch + 1}/{epochs}, Train AUC: {train_auc:.4f}')
    

    # 保存模型和数据集
    joblib.dump(model, output_model_path)
    with open(output_dataset_path, 'wb') as f:
        pickle.dump(dataset, f)


