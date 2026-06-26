from src.library_assistant import LibraryAssistant
import os
import sys

def main():
    """
    Main entry point for the Smart Library Assistant application
    """
    print("🚀 Starting Smart Library Assistant...")
    print("=" * 50)
    
    # Get the absolute path to the CSV file
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'library_books.csv')
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        print("📁 Please ensure the CSV file is in the correct location.")
        print("   Expected path: data/library_books.csv")
        print("\n📝 You can create the CSV file with the following columns:")
        print("   book_id,title,author,genre,year,copies,available,category,rating")
        return
    
    try:
        # Initialize the library assistant
        print("📚 Initializing Library Assistant...")
        assistant = LibraryAssistant(data_path)
        
        # Check if data was loaded successfully
        if assistant.data is None:
            print("❌ Failed to load data. Please check the CSV file.")
            return
        
        # Start the interactive menu
        print("✅ Ready to assist you!")
        print("=" * 50)
        assistant.interactive_menu()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        print("📁 Please make sure the file exists and is accessible.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        print("🔄 Please check your data file and try again.")

if __name__ == "__main__":
    main()