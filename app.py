from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import numpy as np
from movie_feature import load_movie_features
from user_feature import build_user_feature_list
from tqdm import tqdm
import csv
from imdb_crawler import IMDBCrawler
import random
import os
import json

app = Flask(__name__)
# 初始化爬虫，设置5个并发线程
imdb_crawler = IMDBCrawler(max_workers=5, request_delay=0.5)

# 读取映射表
links_csv_path = '/Users/zzcnb123456/AI_project/recommond/ml-32m/links.csv'
movieid_to_imdbid = {}
with open(links_csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        movieid_to_imdbid[int(row['movieId'])] = row['imdbId']

# 加载模型等资源
dataset_path = './model/dataset.pkl'
model_path = './model/model.pkl'
movie_csv = '/Users/zzcnb123456/AI_project/recommond/ml-32m/movies.csv'

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
model = joblib.load(model_path)
movie_features = load_movie_features(csv_path=movie_csv)
item_features = dataset.build_item_features(
    ((movie_id, features) for movie_id, features in movie_features['feature_list'].items())
)

@app.route('/')
def index():
    return render_template('select_genres.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    liked_genres = data['liked_genres']
    preferred_year_range = tuple(data['preferred_year_range'])

    query_features = build_user_feature_list(liked_genres, preferred_year_range)

    # 获取符合特征的电影
    # 首先尝试使用交集
    movies_with_features = movie_features[
        movie_features['feature_list'].apply(lambda x: all(feat in x for feat in query_features))
    ]
    movie_ids_with_feature = list(movies_with_features.index)

    # 如果交集为空，则使用并集
    if len(movie_ids_with_feature) == 0:
        print("使用特征交集筛选结果为空，改用并集筛选...")
        movies_with_features = movie_features[
            movie_features['feature_list'].apply(lambda x: any(feat in x for feat in query_features))
        ]
        movie_ids_with_feature = list(movies_with_features.index)

    # 从符合条件的电影中随机选择3个
    random_selected_movies = []
    if len(movie_ids_with_feature) > 3:
        random_selected_movies = random.sample(movie_ids_with_feature, 3)
    else:
        random_selected_movies = movie_ids_with_feature

    # 排除已随机选择的电影，剩余的参与评分排名
    remaining_movies = list(set(movie_ids_with_feature) - set(random_selected_movies))
    

    # 限制剩余电影数量为10000
    if len(remaining_movies) > 1000:
        remaining_movies = random.sample(remaining_movies, 1000)

    movie_id_to_internal = dataset.mapping()[2]
    internal_movie_ids = [movie_id_to_internal[mid] for mid in remaining_movies if mid in movie_id_to_internal]

    # 随机选择50000个用户
    user_id_to_internal = dataset.mapping()[0]
    all_internal_user_ids = list(user_id_to_internal.values())
    if len(all_internal_user_ids) > 50000:
        selected_user_ids = random.sample(all_internal_user_ids, 50000)
    else:
        selected_user_ids = all_internal_user_ids

    print(f"Processing {len(selected_user_ids)} users for {len(internal_movie_ids)} movies...")

    # 计算用户平均分数
    user_scores = []
    batch_size = 100
    for i in range(0, len(selected_user_ids), batch_size):
        batch_users = selected_user_ids[i:i+batch_size]
        for uid in batch_users:
            scores = model.predict(
                user_ids=np.repeat(uid, len(internal_movie_ids)),
                item_ids=np.array(internal_movie_ids),
                item_features=item_features
            )
            avg_score = np.mean(scores)
            user_scores.append((uid, avg_score))

    # 选择前1000个最高分用户
    top_users = sorted(user_scores, key=lambda x: -x[1])[:1000]
    top_user_internal_ids = [u[0] for u in top_users]

    print("Calculating final recommendations...")

    # 为这1000个用户计算所有电影的得分
    all_item_internal_ids = list(movie_id_to_internal.values())
    internal_to_movie_id = {v: k for k, v in movie_id_to_internal.items()}

    movie_score_sum = np.zeros(len(all_item_internal_ids))
    batch_size = 10
    for i in range(0, len(top_user_internal_ids), batch_size):
        batch_users = top_user_internal_ids[i:i+batch_size]
        for uid in batch_users:
            scores = model.predict(
                user_ids=np.repeat(uid, len(all_item_internal_ids)),
                item_ids=np.array(all_item_internal_ids),
                item_features=item_features
            )
            movie_score_sum += scores

    movie_score_avg = movie_score_sum / len(top_user_internal_ids)
    # 选择评分最高的7部电影
    top_7_indices = np.argsort(-movie_score_avg)[:7]
    top_7_movie_ids = [internal_to_movie_id[i] for i in top_7_indices]
    top_7_movie_ids = [int(mid) for mid in top_7_movie_ids]

    # 合并随机选择的3部和评分最高的7部电影，并打乱顺序
    final_movie_ids = random_selected_movies + top_7_movie_ids
    random.shuffle(final_movie_ids)
    
    print("Fetching movie details using IMDB crawler...")

    # 获取所有电影的IMDb ID
    imdb_ids = []
    for mid in final_movie_ids:
        imdb_id = movieid_to_imdbid.get(mid, '')
        if imdb_id:
            imdb_ids.append(imdb_id)

    # 使用IMDB爬虫批量获取电影信息
    movie_info_dict = imdb_crawler.batch_crawl(imdb_ids)

    # 整理结果
    results = []
    for mid in final_movie_ids:
        imdb_id = movieid_to_imdbid.get(mid, '')
        if imdb_id and imdb_id in movie_info_dict:
            movie_info = movie_info_dict[imdb_id]
            # 标记电影来源（随机选择还是评分推荐）
            movie_info['selection_method'] = 'random' if mid in random_selected_movies else 'rating'
            results.append({
                'movieId': mid,
                'imdbId': imdb_id,
                'info': movie_info
            })

    return jsonify({'recommended_movies': results})

if __name__ == '__main__':
    app.run(debug=True)
