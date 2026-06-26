import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize the DataLoader with the CSV file path
        
        Args:
            file_path (str): Path to the CSV file containing book data
        """
        self.file_path = file_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load CSV data into pandas DataFrame"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully! {len(self.data)} books found.")
            print(f"📊 Columns: {', '.join(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: File '{self.file_path}' not found.")
            print("Please ensure the CSV file is in the correct location.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def get_book_by_id(self, book_id):
        """
        Get book details by ID
        
        Args:
            book_id (int): The ID of the book to retrieve
        
        Returns:
            DataFrame: Book details if found, empty DataFrame otherwise
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['book_id'] == book_id]
    
    def search_books(self, keyword, column='title'):
        """
        Search books by keyword in specified column
        
        Args:
            keyword (str): The keyword to search for
            column (str): The column to search in (title, author, genre, etc.)
        
        Returns:
            DataFrame: Matching books
        """
        if self.data is None:
            return pd.DataFrame()
        
        if column not in self.data.columns:
            print(f"❌ Column '{column}' not found")
            print(f"Available columns: {', '.join(self.data.columns)}")
            return pd.DataFrame()
        
        results = self.data[self.data[column].str.contains(keyword, case=False, na=False)]
        return results
    
    def get_available_books(self):
        """Get all available books (available copies > 0)"""
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['available'] > 0]
    
    def get_unavailable_books(self):
        """Get all unavailable books (available copies = 0)"""
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['available'] == 0]
    
    def get_genre_list(self):
        """Get unique genres in the library"""
        if self.data is None:
            return []
        return self.data['genre'].unique().tolist()
    
    def filter_by_genre(self, genre):
        """
        Filter books by genre
        
        Args:
            genre (str): The genre to filter by
        
        Returns:
            DataFrame: Books in the specified genre
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['genre'].str.lower() == genre.lower()]
    
    def filter_by_rating(self, min_rating=4.0):
        """
        Filter books by minimum rating
        
        Args:
            min_rating (float): Minimum rating threshold
        
        Returns:
            DataFrame: Books with rating >= min_rating
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['rating'] >= min_rating]
    
    def update_availability(self, book_id, change):
        """
        Update book availability (borrow: -1, return: +1)
        
        Args:
            book_id (int): ID of the book to update
            change (int): Change in availability (-1 for borrow, +1 for return)
        
        Returns:
            bool: True if update successful, False otherwise
        """
        if self.data is None:
            return False
        
        idx = self.data[self.data['book_id'] == book_id].index
        if len(idx) > 0:
            # Check if enough copies available for borrowing
            if change < 0 and self.data.loc[idx, 'available'].iloc[0] + change < 0:
                print("❌ Not enough copies available")
                return False
            
            self.data.loc[idx, 'available'] += change
            print(f"✅ Availability updated: {self.data.loc[idx, 'available'].iloc[0]} copies now available")
            return True
        return False
    
    def get_book_stats(self):
        """
        Get basic statistics about the collection
        
        Returns:
            dict: Dictionary containing various statistics
        """
        if self.data is None:
            return {}
        
        return {
            'total_books': len(self.data),
            'total_copies': self.data['copies'].sum(),
            'available_copies': self.data['available'].sum(),
            'unavailable_copies': self.data['copies'].sum() - self.data['available'].sum(),
            'avg_rating': self.data['rating'].mean(),
            'unique_genres': len(self.data['genre'].unique()),
            'genres': self.data['genre'].value_counts().to_dict(),
            'authors': self.data['author'].nunique(),
            'year_range': f"{self.data['year'].min()} - {self.data['year'].max()}"
        }
    
    def get_books_by_author(self, author):
        """
        Get all books by a specific author
        
        Args:
            author (str): Author name to search for
        
        Returns:
            DataFrame: Books by the specified author
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data[self.data['author'].str.contains(author, case=False, na=False)]
    
    def get_top_rated_books(self, n=10):
        """
        Get top N highest rated books
        
        Args:
            n (int): Number of books to return
        
        Returns:
            DataFrame: Top rated books
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data.nlargest(n, 'rating')
    
    def get_most_available(self, n=10):
        """
        Get books with most available copies
        
        Args:
            n (int): Number of books to return
        
        Returns:
            DataFrame: Books with most available copies
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data.nlargest(n, 'available')
    
    def save_data(self, file_path=None):
        """
        Save the current data back to CSV
        
        Args:
            file_path (str): Path to save the file (defaults to original path)
        """
        if self.data is None:
            print("❌ No data to save")
            return
        
        save_path = file_path if file_path else self.file_path
        self.data.to_csv(save_path, index=False)
        print(f"✅ Data saved to {save_path}")