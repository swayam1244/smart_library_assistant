import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class BookRecommender:
    def __init__(self, data):
        """
        Initialize the BookRecommender with book data
        
        Args:
            data (DataFrame): Book data containing title, author, genre, etc.
        """
        self.data = data.copy() if data is not None else None
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.feature_names = []
        
        if self.data is not None:
            self.build_similarity_matrix()
        else:
            print("❌ No data provided for recommendation system")
    
    def build_similarity_matrix(self):
        """
        Build TF-IDF similarity matrix based on book features
        Uses content-based filtering with multiple features
        """
        print("🔄 Building recommendation system...")
        
        # Combine relevant features for content-based filtering
        self.data['combined_features'] = (
            self.data['title'].fillna('') + ' ' +
            self.data['author'].fillna('') + ' ' +
            self.data['genre'].fillna('') + ' ' +
            self.data['category'].fillna('') + ' ' +
            self.data['year'].astype(str) + ' ' +
            self.data['rating'].astype(str)
        )
        
        # Create TF-IDF matrix
        print("📊 Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_features'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate cosine similarity
        print("🔢 Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print("✅ Recommendation system initialized successfully!")
        print(f"   - {len(self.data)} books indexed")
        print(f"   - {len(self.feature_names)} features extracted")
    
    def get_recommendations(self, book_id, top_n=5):
        """
        Get top N book recommendations based on similarity
        
        Args:
            book_id (int): ID of the book to get recommendations for
            top_n (int): Number of recommendations to return
        
        Returns:
            DataFrame: Recommended books with similarity scores
        """
        if self.data is None or self.similarity_matrix is None:
            print("❌ Recommendation system not initialized")
            return pd.DataFrame()
        
        if book_id not in self.data['book_id'].values:
            print(f"❌ Book ID {book_id} not found")
            return pd.DataFrame()
        
        idx = self.data[self.data['book_id'] == book_id].index[0]
        
        # Get similarity scores for the book
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar books (excluding the book itself)
        sim_scores = sim_scores[1:top_n+1]
        
        if not sim_scores:
            print("❌ No recommendations found")
            return pd.DataFrame()
        
        # Get book indices and similarity scores
        book_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Get recommendations
        recommendations = self.data.iloc[book_indices][
            ['book_id', 'title', 'author', 'genre', 'rating', 'available']
        ].copy()
        recommendations['similarity_score'] = similarity_scores
        recommendations['similarity_percentage'] = (similarity_scores * 100).round(2)
        
        return recommendations
    
    def get_recommendations_by_genre(self, genre, top_n=5):
        """
        Get top N books in a specific genre
        
        Args:
            genre (str): The genre to get recommendations for
            top_n (int): Number of books to return
        
        Returns:
            DataFrame: Top books in the genre
        """
        if self.data is None:
            print("❌ No data available")
            return pd.DataFrame()
        
        genre_books = self.data[self.data['genre'].str.lower() == genre.lower()]
        
        if len(genre_books) == 0:
            print(f"❌ No books found in genre '{genre}'")
            return pd.DataFrame()
        
        # Sort by rating and availability
        genre_books = genre_books.sort_values(
            ['rating', 'available'], 
            ascending=[False, False]
        )
        
        return genre_books.head(top_n)[
            ['book_id', 'title', 'author', 'rating', 'available']
        ]
    
    def get_highly_rated_books(self, min_rating=4.5, top_n=10):
        """
        Get highly rated books
        
        Args:
            min_rating (float): Minimum rating threshold
            top_n (int): Number of books to return
        
        Returns:
            DataFrame: Highly rated books
        """
        if self.data is None:
            print("❌ No data available")
            return pd.DataFrame()
        
        high_rated = self.data[self.data['rating'] >= min_rating]
        if len(high_rated) == 0:
            print(f"❌ No books found with rating >= {min_rating}")
            return pd.DataFrame()
        
        high_rated = high_rated.sort_values('rating', ascending=False)
        return high_rated.head(top_n)[
            ['book_id', 'title', 'author', 'genre', 'rating', 'available']
        ]
    
    def get_similar_books_by_description(self, description, top_n=5):
        """
        Find books similar to a text description
        
        Args:
            description (str): Text description to search for similar books
            top_n (int): Number of books to return
        
        Returns:
            DataFrame: Books similar to the description
        """
        if self.data is None or self.vectorizer is None:
            print("❌ Recommendation system not initialized")
            return pd.DataFrame()
        
        # Transform the description using the same vectorizer
        desc_vector = self.vectorizer.transform([description])
        
        # Calculate similarity with all books
        similarities = cosine_similarity(desc_vector, self.tfidf_matrix)
        
        # Get top N similar books
        sim_scores = list(enumerate(similarities[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N
        sim_scores = sim_scores[:top_n]
        top_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = self.data.iloc[top_indices][
            ['book_id', 'title', 'author', 'genre', 'rating', 'available']
        ].copy()
        recommendations['similarity_score'] = similarity_scores
        recommendations['similarity_percentage'] = (similarity_scores * 100).round(2)
        
        return recommendations
    
    def get_personalized_recommendations(self, favorite_books, top_n=5):
        """
        Get personalized recommendations based on favorite books
        
        Args:
            favorite_books (list): List of book IDs the user likes
            top_n (int): Number of recommendations to return
        
        Returns:
            DataFrame: Personalized recommendations
        """
        if self.data is None or self.similarity_matrix is None:
            print("❌ Recommendation system not initialized")
            return pd.DataFrame()
        
        # Get average similarity scores from favorite books
        combined_scores = np.zeros(len(self.data))
        valid_books = 0
        
        for book_id in favorite_books:
            if book_id in self.data['book_id'].values:
                idx = self.data[self.data['book_id'] == book_id].index[0]
                combined_scores += self.similarity_matrix[idx]
                valid_books += 1
        
        if valid_books == 0:
            print("❌ No valid favorite books found")
            return pd.DataFrame()
        
        # Average the scores
        combined_scores = combined_scores / valid_books
        
        # Get indices sorted by similarity (excluding favorite books)
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Filter out favorite books
        favorite_indices = [self.data[self.data['book_id'] == bid].index[0] 
                           for bid in favorite_books if bid in self.data['book_id'].values]
        sorted_indices = [idx for idx in sorted_indices if idx not in favorite_indices]
        
        # Get top recommendations
        top_indices = sorted_indices[:top_n]
        recommendations = self.data.iloc[top_indices][
            ['book_id', 'title', 'author', 'genre', 'rating', 'available']
        ].copy()
        recommendations['similarity_score'] = combined_scores[top_indices]
        recommendations['similarity_percentage'] = (recommendations['similarity_score'] * 100).round(2)
        
        return recommendations
    
    def get_author_recommendations(self, author, top_n=5):
        """
        Get recommendations from same author or similar authors
        
        Args:
            author (str): Author name
            top_n (int): Number of recommendations to return
        
        Returns:
            DataFrame: Books by similar authors
        """
        if self.data is None:
            print("❌ No data available")
            return pd.DataFrame()
        
        # Get books by the same author
        author_books = self.data[self.data['author'].str.contains(author, case=False)]
        
        if len(author_books) > 0:
            return author_books.head(top_n)[
                ['book_id', 'title', 'author', 'genre', 'rating', 'available']
            ]
        
        # If author not found, find similar authors based on genre
        print(f"Author '{author}' not found. Finding similar authors...")
        
        # Find authors with similar genres
        all_authors = self.data['author'].unique()
        results = []
        
        for other_author in all_authors:
            if other_author.lower() != author.lower():
                author_genres = set(self.data[self.data['author'] == other_author]['genre'])
                similarity = len(author_genres.intersection(set(self.data['genre']))) / len(author_genres)
                if similarity > 0.5:
                    books = self.data[self.data['author'] == other_author]
                    results.append((other_author, books))
        
        if results:
            # Sort by number of books
            results.sort(key=lambda x: len(x[1]), reverse=True)
            return results[0][1].head(top_n)[
                ['book_id', 'title', 'author', 'genre', 'rating', 'available']
            ]
        
        return pd.DataFrame()