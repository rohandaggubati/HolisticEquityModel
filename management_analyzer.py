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
            print("SEC API client initialized successfully")
        else:
            print("SEC API key not found. Running in limited mode without SEC data.")
        
        # Download required NLTK data
        print("Downloading NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"Warning: Error downloading NLTK data: {str(e)}")
            print("Continuing with limited NLP capabilities...")
        
        # Enhanced keywords for management analysis
        self.positive_keywords = [
            'innovation', 'growth', 'leadership', 'execution', 'strategy',
            'vision', 'competitive advantage', 'market share', 'efficiency',
            'profitability', 'sustainability', 'research and development',
            'digital transformation', 'artificial intelligence', 'machine learning',
            'blockchain', 'cloud computing', 'data analytics', 'automation',
            'customer experience', 'operational excellence', 'market expansion'
        ]
        
        self.founder_keywords = [
            'founder', 'co-founder', 'founding', 'established by',
            'started by', 'created by', 'founded by', 'founding team',
            'founding member', 'founding partner'
        ]
        
        self.leadership_keywords = [
            'CEO', 'Chief Executive Officer', 'leadership team', 'executive team',
            'management team', 'board of directors', 'senior management',
            'executive leadership', 'corporate governance'
        ]
        
        self.risk_keywords = [
            'risk management', 'compliance', 'regulatory', 'cybersecurity',
            'data privacy', 'internal controls', 'audit committee',
            'risk oversight', 'enterprise risk'
        ]
        
        print("Management Analyzer initialization complete")

    def get_10k_filing(self, ticker, year):
        """Get the most recent 10-K filing for a company with enhanced error handling"""
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
                
                # Try multiple methods to get the filing text
                text_content = None
                
                # Method 1: Direct text from API
                if 'text' in filing:
                    text_content = filing['text']
                    print("Retrieved text directly from API")
                
                # Method 2: Parse from filing URL
                if not text_content and 'linkToFilingDetails' in filing:
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        
                        print(f"Fetching text from filing URL for {ticker}...")
                        url = filing['linkToFilingDetails']
                        response = requests.get(url, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract text from the filing, focusing on relevant sections
                        relevant_sections = []
                        
                        # Look for management discussion section
                        md_a = soup.find(text=lambda text: text and "Management's Discussion" in text)
                        if md_a:
                            relevant_sections.extend(md_a.find_all_next('p', limit=50))
                        
                        # Look for business description
                        business = soup.find(text=lambda text: text and "Business" in text)
                        if business:
                            relevant_sections.extend(business.find_all_next('p', limit=30))
                        
                        # Look for risk factors
                        risks = soup.find(text=lambda text: text and "Risk Factors" in text)
                        if risks:
                            relevant_sections.extend(risks.find_all_next('p', limit=20))
                        
                        text_content = ' '.join([p.get_text() for p in relevant_sections])
                        print("Retrieved and parsed text from filing URL")
                    except Exception as e:
                        print(f"Error parsing filing URL: {str(e)}")
                
                if text_content:
                    return text_content
                    
                print(f"No text content available in filing for {ticker}")
                return None
                
            print(f"No 10-K filing found for {ticker}")
            return None
            
        except Exception as e:
            print(f"Error fetching 10-K for {ticker}: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob with enhanced error handling"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            return {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.5}

    def analyze_management_quality(self, text):
        """Analyze management quality from text using enhanced NLP"""
        if not text:
            print("No text provided for analysis")
            return self._get_default_metrics()
        
        metrics = self._get_default_metrics()
        
        try:
            # Tokenize text into sentences
            sentences = sent_tokenize(text)
            print(f"Tokenized {len(sentences)} sentences")
            
            # Find management-related sentences
            management_sentences = []
            leadership_sentences = []
            risk_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in self.positive_keywords):
                    management_sentences.append(sentence)
                if any(keyword in sentence_lower for keyword in self.leadership_keywords):
                    leadership_sentences.append(sentence)
                if any(keyword in sentence_lower for keyword in self.risk_keywords):
                    risk_sentences.append(sentence)
            
            print(f"Found {len(management_sentences)} management-related sentences")
            print(f"Found {len(leadership_sentences)} leadership-related sentences")
            print(f"Found {len(risk_sentences)} risk-related sentences")
            
            # Analyze sentiment for different types of sentences
            if management_sentences:
                sentiments = [self.analyze_sentiment(sentence) for sentence in management_sentences]
                metrics['management_sentiment'] = np.mean([s['polarity'] for s in sentiments])
                metrics['management_confidence'] = 1 - np.mean([s['subjectivity'] for s in sentiments])
            
            if leadership_sentences:
                sentiments = [self.analyze_sentiment(sentence) for sentence in leadership_sentences]
                metrics['leadership_sentiment'] = np.mean([s['polarity'] for s in sentiments])
            
            if risk_sentences:
                sentiments = [self.analyze_sentiment(sentence) for sentence in risk_sentences]
                metrics['risk_sentiment'] = np.mean([s['polarity'] for s in sentiments])
            
            # Check for founder-led mentions
            metrics['founder_led'] = any(
                keyword in text.lower() for keyword in self.founder_keywords
            )
            
            # Calculate innovation score
            innovation_keywords = [
                'innovation', 'research', 'development', 'technology',
                'artificial intelligence', 'machine learning', 'digital',
                'automation', 'cloud', 'data analytics'
            ]
            metrics['innovation_score'] = sum(
                1 for keyword in innovation_keywords 
                if keyword in text.lower()
            )
            
            # Count strategy mentions
            strategy_keywords = [
                'strategy', 'strategic', 'plan', 'initiative',
                'roadmap', 'vision', 'mission', 'objective'
            ]
            metrics['strategy_mentions'] = sum(
                1 for keyword in strategy_keywords 
                if keyword in text.lower()
            )
            
            # Calculate risk management score
            metrics['risk_management_score'] = len(risk_sentences) / max(len(sentences) * 0.05, 1)
            
            return metrics
            
        except Exception as e:
            print(f"Error in management quality analysis: {str(e)}")
            return metrics

    def _get_default_metrics(self):
        """Return default metrics dictionary"""
        return {
            'management_sentiment': 0.0,
            'management_confidence': 0.5,
            'leadership_sentiment': 0.0,
            'risk_sentiment': 0.0,
            'founder_led': False,
            'innovation_score': 0,
            'strategy_mentions': 0,
            'risk_management_score': 0.0
        }

    def get_management_score(self, ticker, year=2023):
        """Get overall management score for a company"""
        print(f"\nProcessing management score for {ticker}...")
        start_time = time.time()
        
        if not self.use_sec_api:
            print(f"Running in limited mode for {ticker} - using default metrics")
            metrics = self._get_default_metrics()
        else:
            filing_text = self.get_10k_filing(ticker, year)
            if not filing_text:
                print(f"Could not process {ticker} - no filing text available")
                return None
            
            try:
                # Analyze management quality from 10-K
                metrics = self.analyze_management_quality(filing_text)
            except Exception as e:
                print(f"Error analyzing management for {ticker}: {str(e)}")
                return None
        
        # Calculate composite score (0-100)
        composite_score = (
            (metrics['management_sentiment'] + 1) * 15 +    # Scale -1 to 1 to 0-30
            (metrics['management_confidence']) * 10 +       # 0-10 points for confidence
            (metrics['leadership_sentiment'] + 1) * 10 +    # Scale -1 to 1 to 0-20
            (metrics['founder_led'] * 15) +                # 15 points if founder-led
            min(metrics['innovation_score'] * 2, 15) +     # Up to 15 points for innovation
            min(metrics['strategy_mentions'], 5) +         # Up to 5 points for strategy
            min(metrics['risk_management_score'] * 10, 5)  # Up to 5 points for risk management
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