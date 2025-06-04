from lightfm import LightFM
from lightfm.data import Dataset
from scipy import sparse
import numpy as np

import pandas as pd
from lightfm.evaluation import precision_at_k, auc_score
from tqdm import tqdm
import re
import joblib 
import pickle
from movie_feature import load_movie_features
import argparse
from user_feature import build_user_feature_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for movie recommendations.')
    parser.add_argument('--movie_csv', type=str, default='/Users/zzcnb123456/AI_project/recommond/ml-32m/movies.csv',
                        help='Path to the movies CSV file.')
    parser.add_argument('--dataset_path', type=str, default='./model/dataset.pkl',
                        help='Path to the dataset file.')
    parser.add_argument('--model_path', type=str, default='./model/model.pkl',
                        help='Path to the trained model file.')
    args = parser.parse_args()
    movie_csv = args.movie_csv
    dataset_path = args.dataset_path        
    model_path = args.model_path

    liked_genres=['Action', 'Sci-Fi', 'Adventure']
    preferred_year_range=(2000, 2020)  # 偏好2000~2020之间


    # 推理流程：
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    model = joblib.load(model_path)


    movie_features = load_movie_features(csv_path=movie_csv)
    item_features = dataset.build_item_features(
        ((movie_id, features) for movie_id, features in movie_features['feature_list'].items())
    )

    # 查询某电影特征，比如 'Sci-Fi'
    query_features = build_user_feature_list(liked_genres,preferred_year_range) 

    # 找出带有该特征的电影
    movies_with_features = movie_features[
        movie_features['feature_list'].apply(lambda x: all(feat in x for feat in query_features))
    ]
    movie_ids_with_feature = list(movies_with_features.index)

    movie_id_to_internal = dataset.mapping()[2]
    internal_movie_ids = [movie_id_to_internal[mid] for mid in movie_ids_with_feature if mid in movie_id_to_internal]

    user_id_to_internal = dataset.mapping()[0]
    all_internal_user_ids = list(user_id_to_internal.values())

    # 计算所有用户对这些电影的平均预测分数
    user_scores = []
    for uid in tqdm(all_internal_user_ids):
        scores = model.predict(
            user_ids=np.repeat(uid, len(internal_movie_ids)),
            item_ids=np.array(internal_movie_ids),
            item_features=item_features
        )
        avg_score = np.mean(scores)
        user_scores.append((uid, avg_score))

    # 选出前1000个用户
    top_users = sorted(user_scores, key=lambda x: -x[1])[:1000]
    top_user_internal_ids = [u[0] for u in top_users]

    # 找出这些用户最喜欢的电影Top10
    all_item_internal_ids = list(movie_id_to_internal.values())
    internal_to_movie_id = {v:k for k,v in movie_id_to_internal.items()}

    movie_score_sum = np.zeros(len(all_item_internal_ids))
    for uid in tqdm(top_user_internal_ids):
        scores = model.predict(
            user_ids=np.repeat(uid, len(all_item_internal_ids)),
            item_ids=np.array(all_item_internal_ids),
            item_features=item_features
        )
        movie_score_sum += scores

    movie_score_avg = movie_score_sum / len(top_user_internal_ids)
    top_10_indices = np.argsort(-movie_score_avg)[:10]
    top_10_movie_ids = [internal_to_movie_id[i] for i in top_10_indices]

    print("推荐的电影Top10：")
    for mid in top_10_movie_ids:
        print(mid)