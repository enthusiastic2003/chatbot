import pandas as pd
import sqlite3
import os

aspects = [
    "Earnings",
    "Revenue",
    "Margins",
    "Dividend",
    "EBITDA",
    "Debt",
    "Sentiment"
]

aspects_average = [asp+"_average" for asp in aspects] 

class NewsDatabase:
    def __init__(self, db_name1='stock_news.db', db_name2='result_stock_news.db'):
        """Initialize the NewsDatabase with a specified SQLite database name and aspects list."""
        
        db_name1 = os.path.join(os.path.dirname(__file__), db_name1)
        db_name2 = os.path.join(os.path.dirname(__file__), db_name2)

        
        # Connect to SQLite database
        self.connection = sqlite3.connect(db_name1)
        self.cursor = self.connection.cursor()
        self.connection2 = sqlite3.connect(db_name2)
        self.cursor2 = self.connection2.cursor()
        
        # Define the list of aspects
        self.aspects = aspects 
        
        # Initialize the SentimentAnalyser with the specified aspects
        
        # Create the table with columns for each aspect
        self.create_table()
        self.to_dataframe()
        
    def create_table(self):
        """Create a table for news articles with columns for each aspect's score."""
        # Basic schema with fixed columns
        columns = '''
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT,
            headline TEXT,
            published_date TEXT,
            url TEXT,
            embedding BLOB,
        '''
        
        # Add a column for each aspect using the aspect's name directly
        for aspect in range(len(self.aspects)-1):
            columns += f'{self.aspects[aspect]} REAL,\n'
        
        columns += f'{self.aspects[-1]} REAL\n'
        # Create the table if it doesn't already exist
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS news_articles ({columns})
        ''')
        self.connection.commit()
        # print(f"Table created with columns for each aspect with this query: {columns}")

    def insert_data(self, ticker, headline, published_date, url):
        """Insert data into the database, including scores for each aspect."""

        # Generate aspect-based sentiment scores for each aspect
        aspect_scores = self.sentiment_analyser.analyze_sentiment(headline)

        # Prepare SQL for dynamic insertion based on aspects
        aspect_columns = ', '.join(self.aspects)  # Create columns list for SQL
        placeholders = ', '.join(['?'] * (4 + len(self.aspects)))  # Placeholder for SQL
        values = [ticker, headline, published_date, url] + list(aspect_scores.values())
        
        # print(values, type(values))
        # print(aspect_columns, type(aspect_columns))
        # Insert into the database
        self.cursor.execute(f'''
            INSERT INTO news_articles (stock_symbol, headline, published_date, url, {aspect_columns})
            VALUES ({placeholders})
        ''', values)
        self.connection.commit()
        # print(f"Data inserted for {ticker} with aspect scores.")

    def to_dataframe(self):
        """Retrieve all data from the database and return it as a Pandas DataFrame with average columns for each aspect, ignoring 0 values."""
        query = "SELECT * FROM news_articles"
        self.df = pd.read_sql_query(query, self.connection)
        query = "SELECT * FROM stock_news_results"

        self.df2 = pd.read_sql_query(query, self.connection2)

        return self.df,self.df2

    def get_values_by_ticker(self, ticker):
        """
        Retrieve the values for each aspect for the given stock symbol and return them as a flat dictionary.
        """
        # Filter the dataframe for the specified ticker
        ticker_df = self.df2[self.df2['stock_symbol'] == ticker]
        
        # Convert to dictionary and get the first row without indices
        if not ticker_df.empty:
            return ticker_df.iloc[0].to_dict()
        return {}

    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()
        # print("Database connection closed.")
