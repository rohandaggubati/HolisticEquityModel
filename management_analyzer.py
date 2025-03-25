import os
from sec_api import QueryApi
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from dotenv import load_dotenv
import time
from textblob import TextBlob

class ManagementAnalyzer:
    def __init__(self):
        print("Initializing Management Analyzer...")
        load_dotenv()
        self.sec_api_key = os.getenv('SEC_API_KEY')
        self.use_sec_api = bool(self.sec_api_key)
        
        if self.use_sec_api:
            print("Setting up SEC API client...")
            self.query_api = QueryApi(api_key=self.sec_api_key)
        else:
            print("SEC API key not found. Running in limited mode without SEC data.")
        
        # Download required NLTK data
        print("Downloading NLTK data...")
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
        except Exception as e:
            print(f"Warning: Error downloading NLTK data: {str(e)}")
            print("Continuing with limited NLP capabilities...")
        
        # Keywords for management analysis
        self.positive_keywords = [
            'innovation', 'growth', 'leadership', 'execution', 'strategy',
            'vision', 'competitive advantage', 'market share', 'efficiency',
            'profitability', 'sustainability', 'research and development'
        ]
        
        self.founder_keywords = [
            'founder', 'co-founder', 'founding', 'established by',
            'started by', 'created by'
        ]
        print("Management Analyzer initialization complete")

    def get_10k_filing(self, ticker, year):
        """Get the most recent 10-K filing URL for a company"""
        if not self.use_sec_api:
            return None
        
        try:
            print(f"Fetching 10-K filing for {ticker} ({year})...")
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"10-K\" AND filedAt:[{year}-01-01 TO {year}-12-31]"
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            response = self.query_api.get_filings(query)
            if response['total']['value'] > 0:
                print(f"Successfully retrieved 10-K for {ticker}")
                filing = response['filings'][0]
                
                # Try to get the text directly from the API response
                if 'text' in filing:
                    return filing['text']
                
                # If no text available, try to get the raw URL
                if 'linkToFilingDetails' in filing:
                    import requests
                    from bs4 import BeautifulSoup
                    
                    print(f"Fetching text from filing URL for {ticker}...")
                    url = filing['linkToFilingDetails']
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract text from the filing
                    text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
                    return text_content
                    
                print(f"No text content available in filing for {ticker}")
                return None
                
            print(f"No 10-K filing found for {ticker}")
            return None
            
        except Exception as e:
            print(f"Error fetching 10-K for {ticker}: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0

    def analyze_management_quality(self, text):
        """Analyze management quality from text using NLP"""
        if not text:
            print("No text provided for analysis")
            return {
                'management_sentiment': 0.5,
                'founder_led': False,
                'innovation_score': 5,
                'strategy_mentions': 3
            }
        
        metrics = {
            'management_sentiment': 0.5,
            'founder_led': False,
            'innovation_score': 0,
            'strategy_mentions': 0
        }
        
        try:
            # Tokenize text into sentences
            sentences = sent_tokenize(text)
            print(f"Tokenized {len(sentences)} sentences")
            
            # Find management-related sentences
            management_sentences = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in self.positive_keywords):
                    management_sentences.append(sentence)
            
            print(f"Found {len(management_sentences)} management-related sentences")
            
            # Analyze sentiment
            if management_sentences:
                print(f"Running sentiment analysis on {len(management_sentences)} sentences...")
                sentiments = [self.analyze_sentiment(sentence) for sentence in management_sentences]
                metrics['management_sentiment'] = np.mean(sentiments)
            
            # Check for founder-led mentions
            metrics['founder_led'] = any(
                keyword in text.lower() for keyword in self.founder_keywords
            )
            
            # Calculate innovation score
            innovation_keywords = ['innovation', 'research', 'development', 'technology']
            metrics['innovation_score'] = sum(
                1 for keyword in innovation_keywords 
                if keyword in text.lower()
            )
            
            # Count strategy mentions
            strategy_keywords = ['strategy', 'strategic', 'plan', 'initiative']
            metrics['strategy_mentions'] = sum(
                1 for keyword in strategy_keywords 
                if keyword in text.lower()
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error in management quality analysis: {str(e)}")
            return metrics

    def get_management_score(self, ticker, year=2023):
        """Get overall management score for a company"""
        print(f"\nProcessing management score for {ticker}...")
        start_time = time.time()
        
        # If SEC API is not available, return a default score
        if not self.use_sec_api:
            print(f"Running in limited mode for {ticker} - using default metrics")
            metrics = {
                'management_sentiment': 0.5,
                'founder_led': False,
                'innovation_score': 5,
                'strategy_mentions': 3
            }
        else:
            filing_url = self.get_10k_filing(ticker, year)
            if not filing_url:
                print(f"Could not process {ticker} - no filing URL available")
                return None
            
            try:
                # Analyze management quality from 10-K
                metrics = self.analyze_management_quality(filing_url)
            except Exception as e:
                print(f"Error analyzing management for {ticker}: {str(e)}")
                return None
        
        # Calculate composite score (0-100)
        composite_score = (
            (metrics['management_sentiment'] + 1) * 25 +  # Scale -1 to 1 to 0-50
            (metrics['founder_led'] * 20) +              # Add 20 if founder-led
            min(metrics['innovation_score'] * 2, 20) +   # Up to 20 points for innovation
            min(metrics['strategy_mentions'], 10)        # Up to 10 points for strategy
        )
        
        print(f"Completed management analysis for {ticker} in {time.time() - start_time:.2f} seconds")
        return {
            'composite_score': composite_score,
            'metrics': metrics
        }

if __name__ == "__main__":
    print("Running Management Analyzer test...")
    analyzer = ManagementAnalyzer()
    # Example usage
    result = analyzer.get_management_score('NVDA')
    if result:
        print(f"Management Score: {result['composite_score']}")
        print(f"Detailed Metrics: {result['metrics']}")
    print("Test completed") 