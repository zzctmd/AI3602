import requests
from bs4 import BeautifulSoup
import os
import json
import time
import random
from datetime import datetime
import re
import concurrent.futures
from typing import List, Dict, Any

class IMDBCrawler:
    def __init__(self, max_workers=5, request_delay=1.0):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.cache_dir = 'static/movie_cache'
        self.poster_dir = os.path.join(self.cache_dir, 'posters')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.poster_dir, exist_ok=True)
        self.max_workers = max_workers
        self.request_delay = request_delay
        
    def _download_image(self, url: str, imdb_id: str) -> str:
        """下载电影海报"""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                poster_path = os.path.join(self.poster_dir, f'{imdb_id}.jpg')
                with open(poster_path, 'wb') as f:
                    f.write(response.content)
                return f'/static/movie_cache/posters/{imdb_id}.jpg'
        except Exception as e:
            print(f"Error downloading poster for {imdb_id}: {str(e)}")
        return None

    def get_movie_info(self, imdb_id: str) -> Dict[str, Any]:
        """获取单个电影信息"""
        cache_file = os.path.join(self.cache_dir, f'{imdb_id}.json')
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                cache_time = datetime.fromisoformat(cached_data.get('cache_time', '2000-01-01'))
                if (datetime.now() - cache_time).days < 7 and cached_data.get('title') and cached_data.get('year'):
                    return cached_data

        print(f"Fetching data for {imdb_id}")
        url = f'https://www.imdb.com/title/tt{imdb_id}/'
        
        max_retries = 5
        base_delay = 3
        
        for retry in range(max_retries):
            try:
                delay = base_delay * (2 ** retry) + random.uniform(1, 3)
                print(f"Attempt {retry + 1} for {imdb_id}, waiting {delay:.2f} seconds...")
                time.sleep(delay)
                
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                info = {
                    'title': '',
                    'poster_url': '',
                    'local_poster_path': '',
                    'rating': '',
                    'description': '',
                    'genres': [],
                    'year': '',
                    'director': '',
                    'stars': [],
                    'duration': '',
                    'cache_time': datetime.now().isoformat()
                }

                # 获取标题（使用多个选择器和方法）
                title_found = False
                
                # 方法1：使用data-testid属性
                title_elem = soup.find('h1', {'data-testid': 'hero-title-block__title'})
                if title_elem:
                    info['title'] = title_elem.get_text(strip=True)
                    title_found = True
                
                # 方法2：使用特定类名
                if not title_found:
                    title_selectors = [
                        '.TitleHeader__TitleText-sc-1wu6n3d-0',
                        '.hero__title',
                        '.title_wrapper h1',
                        '[class*="title_header"] h1',
                        '[class*="TitleHeader"] h1',
                        '[class*="title"] h1'
                    ]
                    for selector in title_selectors:
                        title_elem = soup.select_one(selector)
                        if title_elem:
                            # 移除年份和其他额外信息
                            title_text = title_elem.get_text(strip=True)
                            # 移除括号中的年份
                            title_text = re.sub(r'\s*\(\d{4}\)\s*$', '', title_text)
                            info['title'] = title_text
                            title_found = True
                            break
                
                # 方法3：查找页面中第一个h1标签
                if not title_found:
                    h1_elem = soup.find('h1')
                    if h1_elem:
                        title_text = h1_elem.get_text(strip=True)
                        title_text = re.sub(r'\s*\(\d{4}\)\s*$', '', title_text)
                        info['title'] = title_text
                        title_found = True

                # 获取年份（使用多个方法）
                year_found = False
                
                # 方法1：从URL中提取年份
                year_pattern = r'/title/tt\d+/\D*(\d{4})'
                year_match = re.search(year_pattern, response.url)
                if year_match:
                    info['year'] = year_match.group(1)
                    year_found = True
                
                # 方法2：使用多个选择器
                if not year_found:
                    year_selectors = [
                        'a[href*="/releaseinfo"]',
                        '.TitleBlockMetaData__ListItemText-sc-12ein40-2',
                        '#titleYear a',
                        '[class*="ReleaseYear"]',
                        '[class*="release_year"]',
                        'span.nobr',
                        'title_wrapper .year_column'
                    ]
                    
                    for selector in year_selectors:
                        year_elems = soup.select(selector)
                        for elem in year_elems:
                            text = elem.get_text(strip=True)
                            year_match = re.search(r'\b(19|20)\d{2}\b', text)
                            if year_match:
                                info['year'] = year_match.group()
                                year_found = True
                                break
                        if year_found:
                            break
                
                # 方法3：在整个页面文本中搜索年份
                if not year_found:
                    # 搜索标题附近的文本
                    if title_found and title_elem:
                        parent = title_elem.parent
                        if parent:
                            text = parent.get_text()
                            year_match = re.search(r'\b(19|20)\d{2}\b', text)
                            if year_match:
                                info['year'] = year_match.group()
                                year_found = True
                
                # 如果仍然没有找到年份，搜索整个页面
                if not year_found:
                    text = soup.get_text()
                    year_matches = re.finditer(r'\b(19|20)\d{2}\b', text)
                    years = []
                    for match in year_matches:
                        years.append(match.group())
                    if years:
                        # 使用最常见的年份
                        from collections import Counter
                        year_counts = Counter(years)
                        info['year'] = year_counts.most_common(1)[0][0]
                        year_found = True

                # 获取海报
                poster_selectors = [
                    'img.ipc-image',
                    'img[data-testid="hero-media__poster"]',
                    'img.poster',
                    'div.poster img',
                    '[class*="Poster"] img',
                    '[class*="poster"] img'
                ]
                
                for selector in poster_selectors:
                    try:
                        poster_elem = soup.select_one(selector)
                        if poster_elem and 'src' in poster_elem.attrs:
                            info['poster_url'] = poster_elem['src']
                            if not info['poster_url'].startswith('http'):
                                info['poster_url'] = 'https://www.imdb.com' + info['poster_url']
                            info['local_poster_path'] = self._download_image(info['poster_url'], imdb_id)
                            break
                    except Exception as e:
                        print(f"Error getting poster for {imdb_id}: {str(e)}")
                        continue

                # 获取评分
                rating_selectors = [
                    '[data-testid="hero-rating-bar__aggregate-rating__score"]',
                    'span.AggregateRatingButton__RatingScore-sc-1ll29m0-1',
                    'div.AggregateRatingButton__Rating-sc-1ll29m0-2',
                    'span[itemprop="ratingValue"]',
                    '.ratingValue strong',
                    '[class*="rating"] span',
                    '[class*="Rating"] span'
                ]
                
                for selector in rating_selectors:
                    try:
                        rating_elem = soup.select_one(selector)
                        if rating_elem:
                            rating_text = rating_elem.get_text(strip=True)
                            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                            if rating_match:
                                info['rating'] = rating_match.group(1)
                                break
                    except Exception as e:
                        print(f"Error getting rating for {imdb_id}: {str(e)}")
                        continue

                # 获取描述
                plot_selectors = [
                    '[data-testid="plot-xl"]',
                    'div.GenresAndPlot__TextContainerBreakpointXL-sc-cum89p-2',
                    'div.Plot__PlotText-sc-1h8mgri-0',
                    'div.summary_text',
                    '[class*="plot"] p',
                    '[class*="Plot"] p',
                    '[class*="synopsis"] p',
                    '[class*="Synopsis"] p'
                ]
                
                for selector in plot_selectors:
                    try:
                        plot_elem = soup.select_one(selector)
                        if plot_elem:
                            plot_text = plot_elem.get_text(strip=True)
                            if plot_text and not plot_text.lower().startswith('add a plot'):
                                info['description'] = plot_text
                                break
                    except Exception as e:
                        print(f"Error getting plot for {imdb_id}: {str(e)}")
                        continue

                # 获取类型
                genre_selectors = [
                    'span.ipc-chip__text',
                    'a.GenresAndPlot__GenreChip-sc-cum89p-3',
                    'div.subtext a[href*="genres"]',
                    '[class*="Genre"] a',
                    '[class*="genre"] a'
                ]
                
                try:
                    for selector in genre_selectors:
                        genre_elems = soup.select(selector)
                        if genre_elems:
                            info['genres'] = [genre.text.strip() for genre in genre_elems[:3]]
                            break
                except Exception as e:
                    print(f"Error getting genres for {imdb_id}: {str(e)}")

                # 获取导演
                director_selectors = [
                    'a[href*="tt_ov_dr"]',
                    '[class*="Director"] a',
                    '[class*="director"] a',
                    'span[itemprop="director"] a'
                ]
                
                try:
                    for selector in director_selectors:
                        director_elems = soup.select(selector)
                        if director_elems:
                            info['director'] = director_elems[0].get_text(strip=True)
                            break
                except Exception as e:
                    print(f"Error getting director for {imdb_id}: {str(e)}")

                # 获取演员
                star_selectors = [
                    'a[href*="tt_ov_st"]',
                    '[class*="Actor"] a',
                    '[class*="actor"] a',
                    'span[itemprop="actor"] a'
                ]
                
                try:
                    for selector in star_selectors:
                        star_elems = soup.select(selector)
                        if star_elems:
                            info['stars'] = [star.get_text(strip=True) for star in star_elems[:3]]
                            break
                except Exception as e:
                    print(f"Error getting stars for {imdb_id}: {str(e)}")

                # 获取时长
                duration_selectors = [
                    '[class*="Duration"]',
                    '[class*="duration"]',
                    'time[datetime]',
                    'span[itemprop="duration"]'
                ]
                
                try:
                    for selector in duration_selectors:
                        duration_elem = soup.select_one(selector)
                        if duration_elem:
                            duration_text = duration_elem.get_text(strip=True)
                            if 'min' in duration_text.lower():
                                info['duration'] = duration_text
                                break
                except Exception as e:
                    print(f"Error getting duration for {imdb_id}: {str(e)}")

                # 验证必要信息是否获取成功
                if not info['title'] or not info['year']:
                    print(f"Missing required info for {imdb_id}: title='{info['title']}', year='{info['year']}'")
                    raise Exception("Failed to get title or year")
                
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                
                return info
                
            except Exception as e:
                print(f"Attempt {retry + 1} failed for {imdb_id}: {str(e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    print(f"All {max_retries} attempts failed for {imdb_id}")
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    return None

    def batch_crawl(self, imdb_ids: List[str], max_retries=5) -> Dict[str, Any]:
        """并发批量爬取电影信息"""
        results = {}
        
        def crawl_with_retry(imdb_id: str) -> tuple:
            """带重试的爬取函数"""
            for attempt in range(max_retries):
                try:
                    info = self.get_movie_info(imdb_id)
                    if info and info.get('title') and info.get('year'):  # 确保关键信息存在
                        return imdb_id, info
                    time.sleep(random.uniform(2, 5))  # 增加随机延迟
                except Exception as e:
                    print(f"Batch attempt {attempt + 1} failed for {imdb_id}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(3, 7))  # 失败后增加更长的延迟
            return imdb_id, None

        # 使用更保守的并发数
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_id = {executor.submit(crawl_with_retry, imdb_id): imdb_id for imdb_id in imdb_ids}
            
            for future in concurrent.futures.as_completed(future_to_id):
                imdb_id = future_to_id[future]
                try:
                    imdb_id, info = future.result()
                    if info:
                        results[imdb_id] = info
                except Exception as e:
                    print(f"Error processing {imdb_id}: {str(e)}")

        return results

def main():
    """测试爬虫功能"""
    crawler = IMDBCrawler(max_workers=10, request_delay=1.0)
    
    # 测试单个电影爬取
    test_imdb_id = "0114709"  # Toy Story的IMDB ID
    print("Testing single movie crawl...")
    info = crawler.get_movie_info(test_imdb_id)
    if info:
        print(f"Successfully crawled: {info['title']}")
        print(f"Rating: {info['rating']}")
        print(f"Year: {info['year']}")
        print(f"Genres: {', '.join(info['genres'])}")
        print(f"Description: {info['description'][:100]}...")
    
    # 测试并发批量爬取
    test_imdb_ids = ["0114709", "0113497", "0113228", "0114885", "0113041"]
    print("\nTesting batch crawl...")
    results = crawler.batch_crawl(test_imdb_ids)
    print(f"Successfully crawled {len(results)} movies")
    
    # 显示并发爬取结果
    for imdb_id, info in results.items():
        print(f"\nMovie {imdb_id}:")
        print(f"Title: {info['title']}")
        print(f"Rating: {info['rating']}")
        print(f"Year: {info['year']}")

if __name__ == "__main__":
    main() 