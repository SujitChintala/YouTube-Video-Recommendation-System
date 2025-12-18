from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recommendation_system import YouTubeRecommendationSystem
import os

app = Flask(__name__, static_folder='frontend')
CORS(app)

# Initialize recommendation system
print("Initializing Recommendation System...")
recommender = YouTubeRecommendationSystem()
print("Recommendation System Ready")

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('frontend', 'index.html')

@app.route('/api/recommendations/<video_id>')
def get_recommendations(video_id):
    """Get recommendations for a specific video"""
    try:
        n = request.args.get('n', default=10, type=int)
        recommendations = recommender.get_recommendations(video_id=video_id, n_recommendations=n)
        
        if not recommendations:
            return jsonify({'error': 'Video not found'}), 404
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trending')
def get_trending():
    """Get trending videos"""
    try:
        country = request.args.get('country', default=None, type=str)
        category_id = request.args.get('category_id', default=None, type=int)
        n = request.args.get('n', default=20, type=int)
        
        trending = recommender.get_trending_videos(
            country=country,
            category_id=category_id,
            n_videos=n
        )
        
        return jsonify({
            'success': True,
            'videos': trending
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search_videos():
    """Search for videos"""
    try:
        query = request.args.get('q', default='', type=str)
        n = request.args.get('n', default=20, type=int)
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        results = recommender.search_videos(query=query, n_results=n)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<video_id>')
def get_video_details(video_id):
    """Get details for a specific video"""
    try:
        video = recommender.get_video_details(video_id)
        
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        return jsonify({
            'success': True,
            'video': video
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get general statistics"""
    try:
        df = recommender.df
        stats = {
            'total_videos': len(df),
            'total_views': int(df['views'].sum()),
            'total_likes': int(df['likes'].sum()),
            'total_comments': int(df['comment_count'].sum()),
            'countries': df['country'].value_counts().to_dict(),
            'categories': df['category_id'].value_counts().head(10).to_dict()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
