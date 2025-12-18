import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

class YouTubeRecommendationSystem:
    def __init__(self, data_path='data/merged_youtube_data.csv'):
        """Initialize the recommendation engine with data"""
        print("Loading data...")
        self.df = pd.read_csv(data_path)
        self.df = self.df.drop_duplicates(subset=['video_id'], keep='first')
        self._preprocess_data()
        self._build_recommendation_model()
        
    def _preprocess_data(self):
        """Preprocess and clean the data"""
        print("Preprocessing data...")
        
        # Handle missing values
        self.df['tags'] = self.df['tags'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        self.df['title'] = self.df['title'].fillna('')
        
        # Clean tags (remove pipes)
        self.df['tags'] = self.df['tags'].apply(lambda x: x.replace('|', ' ') if isinstance(x, str) else '')
        
        # Create combined features for content-based filtering
        self.df['combined_features'] = (
            self.df['title'] + ' ' + 
            self.df['tags'] + ' ' + 
            self.df['channel_title'] + ' ' +
            self.df['description'].apply(lambda x: x[:200] if isinstance(x, str) else '')
        )
        
        # Calculate engagement score
        self.df['engagement_score'] = (
            self.df['likes'] + 
            self.df['comment_count'] * 2 - 
            self.df['dislikes']
        ) / (self.df['views'] + 1)
        
        # Calculate popularity score
        self.df['popularity_score'] = (
            np.log1p(self.df['views']) * 0.4 +
            np.log1p(self.df['likes']) * 0.3 +
            np.log1p(self.df['comment_count']) * 0.3
        )
        
    def _build_recommendation_model(self):
        """Build TF-IDF model for content similarity"""
        print("Building recommendation model...")
        
        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform the combined features
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        
        print(f"Model built with {len(self.df)} videos")
        
    def get_recommendations(self, video_id=None, title=None, n_recommendations=10):
        """
        Get video recommendations based on either video_id or title.
        
        Parameters:
        - video_id: YouTube video ID
        - title: Video title (used if video_id not provided)
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended videos with details
        """
        
        # Find the video index
        if video_id:
            idx = self.df[self.df['video_id'] == video_id].index
        elif title:
            idx = self.df[self.df['title'].str.contains(title, case=False, na=False)].index
        else:
            return []
        
        if len(idx) == 0:
            return []
        
        idx = idx[0]
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[idx:idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get similar video indices (excluding the input video)
        similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]
        
        # Create recommendations list
        recommendations = []
        for i in similar_indices:
            video = self.df.iloc[i]
            recommendations.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_title': video['channel_title'],
                'views': int(video['views']),
                'likes': int(video['likes']),
                'dislikes': int(video['dislikes']),
                'comment_count': int(video['comment_count']),
                'category_id': int(video['category_id']),
                'country': video['country'],
                'thumbnail': video['thumbnail_link'],
                'publish_time': video['publish_time'],
                'similarity_score': float(cosine_similarities[i]),
                'popularity_score': float(video['popularity_score'])
            })
        
        return recommendations
    
    def get_trending_videos(self, country=None, category_id=None, n_videos=10):
        """
        Get trending videos based on popularity score
        
        Parameters:
        - country: Filter by country code (CA, GB, IN, US)
        - category_id: Filter by category ID
        - n_videos: Number of videos to return
        
        Returns:
        - List of trending videos
        """
        df_filtered = self.df.copy()
        
        if country:
            df_filtered = df_filtered[df_filtered['country'] == country]
        
        if category_id:
            df_filtered = df_filtered[df_filtered['category_id'] == category_id]
        
        # Sort by popularity score
        trending = df_filtered.nlargest(n_videos, 'popularity_score')
        
        results = []
        for _, video in trending.iterrows():
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_title': video['channel_title'],
                'views': int(video['views']),
                'likes': int(video['likes']),
                'dislikes': int(video['dislikes']),
                'comment_count': int(video['comment_count']),
                'category_id': int(video['category_id']),
                'country': video['country'],
                'thumbnail': video['thumbnail_link'],
                'publish_time': video['publish_time'],
                'popularity_score': float(video['popularity_score'])
            })
        
        return results
    
    def search_videos(self, query, n_results=10):
        """
        Search for videos based on query string
        
        Parameters:
        - query: Search query
        - n_results: Number of results to return
        
        Returns:
        - List of matching videos
        """
        # Search in title and tags
        mask = (
            self.df['title'].str.contains(query, case=False, na=False) |
            self.df['tags'].str.contains(query, case=False, na=False) |
            self.df['channel_title'].str.contains(query, case=False, na=False)
        )
        
        results_df = self.df[mask].nlargest(n_results, 'popularity_score')
        
        results = []
        for _, video in results_df.iterrows():
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_title': video['channel_title'],
                'views': int(video['views']),
                'likes': int(video['likes']),
                'dislikes': int(video['dislikes']),
                'comment_count': int(video['comment_count']),
                'category_id': int(video['category_id']),
                'country': video['country'],
                'thumbnail': video['thumbnail_link'],
                'publish_time': video['publish_time'],
                'popularity_score': float(video['popularity_score'])
            })
        
        return results
    
    def get_video_details(self, video_id):
        """Get details for a specific video"""
        video = self.df[self.df['video_id'] == video_id]
        
        if len(video) == 0:
            return None
        
        video = video.iloc[0]
        return {
            'video_id': video['video_id'],
            'title': video['title'],
            'channel_title': video['channel_title'],
            'views': int(video['views']),
            'likes': int(video['likes']),
            'dislikes': int(video['dislikes']),
            'comment_count': int(video['comment_count']),
            'category_id': int(video['category_id']),
            'country': video['country'],
            'thumbnail': video['thumbnail_link'],
            'publish_time': video['publish_time'],
            'description': video['description'],
            'tags': video['tags'],
            'popularity_score': float(video['popularity_score'])
        }


# Example usage
if __name__ == "__main__":
    # Initialize the recommendation system
    recommender = YouTubeRecommendationSystem()
    
    # Example 1: Get recommendations based on a video ID
    print("\n" + "="*50)
    print("Example 1: Recommendations for a specific video")
    print("="*50)
    video_id = recommender.df.iloc[0]['video_id']
    video_title = recommender.df.iloc[0]['title']
    print(f"\nBased on: {video_title}")
    
    recommendations = recommender.get_recommendations(video_id=video_id, n_recommendations=5)
    print(f"\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Channel: {rec['channel_title']}")
        print(f"   Views: {rec['views']:,} | Likes: {rec['likes']:,}")
        print(f"   Similarity: {rec['similarity_score']:.3f}\n")
    
    # Example 2: Get trending videos
    print("\n" + "="*35)
    print("Example 2: Trending Videos in US")
    print("="*50)
    trending = recommender.get_trending_videos(country='US', n_videos=5)
    for i, video in enumerate(trending, 1):
        print(f"{i}. {video['title']}")
        print(f"   Channel: {video['channel_title']}")
        print(f"   Views: {video['views']:,} | Likes: {video['likes']:,}\n")
    
    # Example 3: Search videos
    print("\n" + "="*35)
    print("Example 3: Search for 'music' videos")
    print("="*50)
    search_results = recommender.search_videos('music', n_results=5)
    for i, video in enumerate(search_results, 1):
        print(f"{i}. {video['title']}")
        print(f"   Channel: {video['channel_title']}")
        print(f"   Views: {video['views']:,}\n")
