from src.data_loader import DataLoader
from src.book_recommendation import BookRecommender
import pandas as pd
import os
import sys

class LibraryAssistant:
    def __init__(self, data_path):
        """
        Initialize the Library Assistant with data
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.data = self.data_loader.data
        
        if self.data is not None:
            self.recommender = BookRecommender(self.data)
            print("\n📚 Welcome to Smart Library Assistant!")
            print(f"📊 Library contains {len(self.data)} books")
        else:
            self.recommender = None
            print("❌ Failed to initialize Library Assistant")
    
    def display_book(self, book):
        """
        Display book information in a formatted way
        
        Args:
            book (Series): A single book record
        """
        if isinstance(book, pd.Series):
            print(f"📚 {book['title']}")
            print(f"   ✍️  Author: {book['author']}")
            print(f"   📖 Genre: {book['genre']}")
            print(f"   📅 Year: {book['year']}")
            print(f"   ⭐ Rating: {book['rating']}/5.0")
            print(f"   📦 Available: {book['available']} copies")
            print(f"   🏷️  Category: {book['category']}")
            print("-" * 50)
        else:
            print("Invalid book data format")
    
    def display_books(self, books, title=""):
        """
        Display multiple books
        
        Args:
            books (DataFrame): Books to display
            title (str): Optional title to display
        """
        if title:
            print(f"\n📌 {title}")
            print("=" * 50)
        
        if books is None or len(books) == 0:
            print("📭 No books found.")
            return
        
        print(f"📊 Showing {len(books)} books:")
        print("-" * 50)
        
        for _, book in books.iterrows():
            self.display_book(book)
    
    def search_books(self, keyword, column='title'):
        """
        Search books and display results
        
        Args:
            keyword (str): Search keyword
            column (str): Column to search in
        
        Returns:
            DataFrame: Search results
        """
        if self.data is None:
            print("❌ No data available")
            return pd.DataFrame()
        
        results = self.data_loader.search_books(keyword, column)
        
        if len(results) == 0:
            print(f"🔍 No books found with '{keyword}' in {column}")
        else:
            self.display_books(results, f"🔍 Search Results for '{keyword}' in {column}")
        
        return results
    
    def recommend_books_by_genre(self, genre):
        """
        Get and display recommendations by genre
        
        Args:
            genre (str): Genre to get recommendations for
        
        Returns:
            DataFrame: Genre recommendations
        """
        if self.recommender is None:
            print("❌ Recommender system not available.")
            return pd.DataFrame()
        
        recommendations = self.recommender.get_recommendations_by_genre(genre)
        if len(recommendations) > 0:
            self.display_books(recommendations, f"🌟 Top Books in {genre}")
        return recommendations
    
    def recommend_similar_books(self, book_id):
        """
        Get and display similar book recommendations
        
        Args:
            book_id (int): Book ID to find similar books for
        
        Returns:
            DataFrame: Similar books recommendations
        """
        if self.recommender is None:
            print("❌ Recommender system not available.")
            return pd.DataFrame()
        
        # Get the book details
        book = self.data_loader.get_book_by_id(book_id)
        if len(book) == 0:
            print(f"❌ Book ID {book_id} not found")
            return pd.DataFrame()
        
        recommendations = self.recommender.get_recommendations(book_id)
        
        if len(recommendations) > 0:
            print(f"\n📚 Books similar to '{book.iloc[0]['title']}':")
            print("=" * 50)
            self.display_books(recommendations)
        else:
            print(f"❌ No similar books found for '{book.iloc[0]['title']}'")
        
        return recommendations
    
    def show_statistics(self):
        """Display library statistics"""
        if self.data is None:
            print("❌ No data available")
            return
        
        stats = self.data_loader.get_book_stats()
        
        print("\n📊 LIBRARY STATISTICS")
        print("=" * 50)
        print(f"📚 Total Books: {stats['total_books']}")
        print(f"📦 Total Copies: {stats['total_copies']}")
        print(f"✅ Available Copies: {stats['available_copies']}")
        print(f"❌ Borrowed Copies: {stats['unavailable_copies']}")
        print(f"⭐ Average Rating: {stats['avg_rating']:.2f}/5.0")
        print(f"📖 Unique Genres: {stats['unique_genres']}")
        print(f"✍️  Unique Authors: {stats['authors']}")
        print(f"📅 Year Range: {stats['year_range']}")
        
        print("\n📂 Books by Genre:")
        print("-" * 30)
        for genre, count in stats['genres'].items():
            percentage = (count / stats['total_books']) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {genre:20} : {count:3} books {bar} {percentage:.1f}%")
    
    def borrow_book(self, book_id):
        """
        Borrow a book from the library
        
        Args:
            book_id (int): ID of the book to borrow
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.data is None:
            print("❌ No data available")
            return False
        
        book = self.data_loader.get_book_by_id(book_id)
        
        if len(book) == 0:
            print(f"❌ Book ID {book_id} not found")
            return False
        
        if book.iloc[0]['available'] <= 0:
            print(f"❌ Sorry, '{book.iloc[0]['title']}' is currently not available.")
            print(f"   All {book.iloc[0]['copies']} copies are borrowed")
            return False
        
        if self.data_loader.update_availability(book_id, -1):
            print(f"✅ Successfully borrowed '{book.iloc[0]['title']}'")
            print(f"   {book.iloc[0]['available'] - 1} copies now available")
            return True
        return False
    
    def return_book(self, book_id):
        """
        Return a book to the library
        
        Args:
            book_id (int): ID of the book to return
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.data is None:
            print("❌ No data available")
            return False
        
        book = self.data_loader.get_book_by_id(book_id)
        
        if len(book) == 0:
            print(f"❌ Book ID {book_id} not found")
            return False
        
        if book.iloc[0]['available'] >= book.iloc[0]['copies']:
            print(f"ℹ️  All copies of '{book.iloc[0]['title']}' are already available.")
            return False
        
        if self.data_loader.update_availability(book_id, 1):
            print(f"✅ Successfully returned '{book.iloc[0]['title']}'")
            print(f"   {book.iloc[0]['available'] + 1} copies now available")
            return True
        return False
    
    def get_book_details(self, book_id):
        """
        Get and display details of a specific book
        
        Args:
            book_id (int): ID of the book
        
        Returns:
            Series: Book details
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        book = self.data_loader.get_book_by_id(book_id)
        if len(book) == 0:
            print(f"❌ Book ID {book_id} not found")
            return None
        
        print(f"\n📖 Book Details:")
        print("=" * 50)
        self.display_book(book.iloc[0])
        return book.iloc[0]
    
    def interactive_menu(self):
        """Interactive menu for the library assistant"""
        if self.data is None:
            print("❌ Cannot start interactive menu - no data loaded")
            return
        
        while True:
            print("\n" + "=" * 50)
            print("📚  SMART LIBRARY ASSISTANT")
            print("=" * 50)
            print("1. 🔍 Search Books")
            print("2. 📖 Get Book Recommendations by Genre")
            print("3. 🔗 Get Similar Books")
            print("4. ⭐ View Highly Rated Books")
            print("5. 📥 Borrow a Book")
            print("6. 📤 Return a Book")
            print("7. 📊 View Library Statistics")
            print("8. 📚 View All Books")
            print("9. ℹ️  Get Book Details")
            print("10. 🚪 Exit")
            print("=" * 50)
            
            choice = input("\n👉 Enter your choice (1-10): ").strip()
            
            if choice == '1':
                keyword = input("🔍 Enter search keyword: ").strip()
                if not keyword:
                    print("❌ Please enter a search keyword")
                    continue
                column = input("🔍 Search by (title/author/genre): ").lower().strip()
                if column not in ['title', 'author', 'genre']:
                    column = 'title'
                    print("ℹ️  Defaulting to 'title' search")
                self.search_books(keyword, column)
                
            elif choice == '2':
                genre = input("📖 Enter genre (e.g., Fiction, Science Fiction, Fantasy): ").strip()
                if not genre:
                    print("❌ Please enter a genre")
                    continue
                self.recommend_books_by_genre(genre)
                
            elif choice == '3':
                try:
                    book_id = int(input("🔗 Enter book ID: ").strip())
                    self.recommend_similar_books(book_id)
                except ValueError:
                    print("❌ Please enter a valid book ID")
                    
            elif choice == '4':
                min_rating = 4.0
                try:
                    rating_input = input("⭐ Enter minimum rating (default 4.0): ").strip()
                    if rating_input:
                        min_rating = float(rating_input)
                except ValueError:
                    print("⚠️  Invalid rating. Using default 4.0")
                
                if self.recommender:
                    highly_rated = self.recommender.get_highly_rated_books(min_rating)
                    if len(highly_rated) > 0:
                        self.display_books(highly_rated, f"⭐ Books with Rating >= {min_rating}")
                    else:
                        print(f"❌ No books found with rating >= {min_rating}")
                else:
                    print("❌ Recommender system not available")
                    
            elif choice == '5':
                try:
                    book_id = int(input("📥 Enter book ID to borrow: ").strip())
                    self.borrow_book(book_id)
                except ValueError:
                    print("❌ Please enter a valid book ID")
                    
            elif choice == '6':
                try:
                    book_id = int(input("📤 Enter book ID to return: ").strip())
                    self.return_book(book_id)
                except ValueError:
                    print("❌ Please enter a valid book ID")
                    
            elif choice == '7':
                self.show_statistics()
                
            elif choice == '8':
                self.display_books(self.data, "📚 All Books in Library")
                
            elif choice == '9':
                try:
                    book_id = int(input("ℹ️  Enter book ID: ").strip())
                    self.get_book_details(book_id)
                except ValueError:
                    print("❌ Please enter a valid book ID")
                
            elif choice == '10':
                print("\n👋 Thank you for using the Smart Library Assistant!")
                print("📚 Happy Reading!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\n⏎ Press Enter to continue...")