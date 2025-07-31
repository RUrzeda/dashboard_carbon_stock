#!/usr/bin/env python3
"""
Comprehensive Bibliometric Analysis for Carbon Stock Modeling Research
Focus: Multisensor modeling of carbon stocks in restoration areas with remote sensing and AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BibliometricAnalyzer:
    def __init__(self, csv_path):
        """Initialize the analyzer with the CSV file"""
        self.df = pd.read_csv(csv_path)
        self.df_filtered = None
        self.brazil_df = None
        self.international_df = None
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        print("Loading and cleaning data...")
        print(f"Original dataset shape: {self.df.shape}")
        
        # Filter years (exclude 2014, work with 2015-2025)
        self.df_filtered = self.df[self.df['PY'].between(2015, 2025)].copy()
        print(f"Filtered dataset shape (2015-2025): {self.df_filtered.shape}")
        
        # Clean and process key columns
        self.df_filtered['AU_clean'] = self.df_filtered['AU'].fillna('')
        self.df_filtered['TI_clean'] = self.df_filtered['TI'].fillna('')
        self.df_filtered['AB_clean'] = self.df_filtered['AB'].fillna('')
        self.df_filtered['KW_clean'] = self.df_filtered['KW_Merged'].fillna('')
        self.df_filtered['SO_clean'] = self.df_filtered['SO'].fillna('')
        self.df_filtered['C1_clean'] = self.df_filtered['C1'].fillna('')
        
        # Identify Brazilian studies
        brazil_keywords = ['BRAZIL', 'BRAZILIAN', 'BRASIL', 'AMAZONIA', 'AMAZON', 'CERRADO', 'CAATINGA', 'MATA ATLANTICA']
        self.df_filtered['is_brazilian'] = (
            self.df_filtered['C1_clean'].str.contains('BRAZIL', case=False, na=False) |
            self.df_filtered['AB_clean'].str.contains('|'.join(brazil_keywords), case=False, na=False) |
            self.df_filtered['TI_clean'].str.contains('|'.join(brazil_keywords), case=False, na=False) |
            self.df_filtered['KW_clean'].str.contains('|'.join(brazil_keywords), case=False, na=False)
        )
        
        self.brazil_df = self.df_filtered[self.df_filtered['is_brazilian']].copy()
        self.international_df = self.df_filtered[~self.df_filtered['is_brazilian']].copy()
        
        print(f"Brazilian studies: {len(self.brazil_df)}")
        print(f"International studies: {len(self.international_df)}")
        
        return self.df_filtered
    
    def extract_authors(self, author_string):
        """Extract individual authors from author string"""
        if pd.isna(author_string) or author_string == '':
            return []
        authors = [author.strip() for author in author_string.split(';')]
        return [author for author in authors if author]
    
    def extract_keywords(self, keyword_string):
        """Extract individual keywords from keyword string"""
        if pd.isna(keyword_string) or keyword_string == '':
            return []
        keywords = [kw.strip().upper() for kw in keyword_string.split(';')]
        return [kw for kw in keywords if kw and len(kw) > 2]
    
    def analyze_authors(self):
        """Analyze most relevant authors"""
        print("Analyzing authors...")
        
        # Extract all authors
        all_authors = []
        for authors_str in self.df_filtered['AU_clean']:
            all_authors.extend(self.extract_authors(authors_str))
        
        author_counts = Counter(all_authors)
        top_authors = pd.DataFrame(author_counts.most_common(20), 
                                 columns=['Author', 'Publications'])
        
        # Brazilian authors
        brazil_authors = []
        for authors_str in self.brazil_df['AU_clean']:
            brazil_authors.extend(self.extract_authors(authors_str))
        
        brazil_author_counts = Counter(brazil_authors)
        top_brazil_authors = pd.DataFrame(brazil_author_counts.most_common(15), 
                                        columns=['Author', 'Publications'])
        
        return top_authors, top_brazil_authors
    
    def analyze_journals(self):
        """Analyze journals with highest number of publications"""
        print("Analyzing journals...")
        
        journal_counts = self.df_filtered['SO_clean'].value_counts().head(20)
        top_journals = pd.DataFrame({
            'Journal': journal_counts.index,
            'Publications': journal_counts.values
        })
        
        # Brazilian studies journals
        brazil_journal_counts = self.brazil_df['SO_clean'].value_counts().head(15)
        top_brazil_journals = pd.DataFrame({
            'Journal': brazil_journal_counts.index,
            'Publications': brazil_journal_counts.values
        })
        
        return top_journals, top_brazil_journals
    
    def analyze_keywords(self):
        """Analyze most frequent keywords"""
        print("Analyzing keywords...")
        
        # Extract all keywords
        all_keywords = []
        for kw_str in self.df_filtered['KW_clean']:
            all_keywords.extend(self.extract_keywords(kw_str))
        
        keyword_counts = Counter(all_keywords)
        top_keywords = pd.DataFrame(keyword_counts.most_common(30), 
                                  columns=['Keyword', 'Frequency'])
        
        # Brazilian keywords
        brazil_keywords = []
        for kw_str in self.brazil_df['KW_clean']:
            brazil_keywords.extend(self.extract_keywords(kw_str))
        
        brazil_keyword_counts = Counter(brazil_keywords)
        top_brazil_keywords = pd.DataFrame(brazil_keyword_counts.most_common(25), 
                                         columns=['Keyword', 'Frequency'])
        
        return top_keywords, top_brazil_keywords
    
    def analyze_temporal_evolution(self):
        """Analyze temporal evolution of publications"""
        print("Analyzing temporal evolution...")
        
        # Overall temporal evolution
        yearly_counts = self.df_filtered['PY'].value_counts().sort_index()
        
        # Brazilian vs International
        brazil_yearly = self.brazil_df['PY'].value_counts().sort_index()
        intl_yearly = self.international_df['PY'].value_counts().sort_index()
        
        # Combine into DataFrame
        years = range(2015, 2026)
        temporal_df = pd.DataFrame({
            'Year': years,
            'Total': [yearly_counts.get(year, 0) for year in years],
            'Brazilian': [brazil_yearly.get(year, 0) for year in years],
            'International': [intl_yearly.get(year, 0) for year in years]
        })
        
        return temporal_df
    
    def extract_countries_from_affiliations(self, affiliation_string):
        """Extract countries from affiliation string"""
        if pd.isna(affiliation_string) or affiliation_string == '':
            return []
        
        # Common country patterns in affiliations
        country_patterns = {
            'USA': ['USA', 'UNITED STATES', 'AMERICA'],
            'CHINA': ['CHINA', 'PEOPLES R CHINA'],
            'BRAZIL': ['BRAZIL', 'BRASIL'],
            'GERMANY': ['GERMANY', 'DEUTSCHLAND'],
            'FRANCE': ['FRANCE'],
            'ITALY': ['ITALY', 'ITALIA'],
            'SPAIN': ['SPAIN', 'ESPANA'],
            'CANADA': ['CANADA'],
            'AUSTRALIA': ['AUSTRALIA'],
            'UK': ['ENGLAND', 'UNITED KINGDOM', 'UK', 'SCOTLAND', 'WALES'],
            'INDIA': ['INDIA'],
            'JAPAN': ['JAPAN'],
            'SOUTH KOREA': ['SOUTH KOREA', 'KOREA'],
            'NETHERLANDS': ['NETHERLANDS', 'HOLLAND'],
            'SWEDEN': ['SWEDEN'],
            'NORWAY': ['NORWAY'],
            'FINLAND': ['FINLAND'],
            'DENMARK': ['DENMARK'],
            'SWITZERLAND': ['SWITZERLAND'],
            'AUSTRIA': ['AUSTRIA'],
            'BELGIUM': ['BELGIUM'],
            'PORTUGAL': ['PORTUGAL'],
            'MEXICO': ['MEXICO'],
            'ARGENTINA': ['ARGENTINA'],
            'CHILE': ['CHILE'],
            'COLOMBIA': ['COLOMBIA'],
            'PERU': ['PERU'],
            'ECUADOR': ['ECUADOR'],
            'VENEZUELA': ['VENEZUELA']
        }
        
        countries = []
        affiliation_upper = affiliation_string.upper()
        
        for country, patterns in country_patterns.items():
            for pattern in patterns:
                if pattern in affiliation_upper:
                    countries.append(country)
                    break
        
        return list(set(countries))  # Remove duplicates
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of publications"""
        print("Analyzing geographic distribution...")
        
        # Extract countries from affiliations
        all_countries = []
        for affiliation in self.df_filtered['C1_clean']:
            all_countries.extend(self.extract_countries_from_affiliations(affiliation))
        
        country_counts = Counter(all_countries)
        geographic_df = pd.DataFrame(country_counts.most_common(20), 
                                   columns=['Country', 'Publications'])
        
        return geographic_df
    
    def identify_ai_techniques(self, text):
        """Identify AI techniques mentioned in text"""
        if pd.isna(text):
            return []
        
        text_upper = text.upper()
        techniques = []
        
        ai_patterns = {
            'RANDOM FOREST': ['RANDOM FOREST', 'RF'],
            'SUPPORT VECTOR MACHINE': ['SUPPORT VECTOR MACHINE', 'SVM'],
            'NEURAL NETWORK': ['NEURAL NETWORK', 'ARTIFICIAL NEURAL NETWORK', 'ANN', 'BP NEURAL', 'CNN', 'CONVOLUTIONAL NEURAL'],
            'DEEP LEARNING': ['DEEP LEARNING', 'DEEP NEURAL'],
            'MACHINE LEARNING': ['MACHINE LEARNING', 'ML'],
            'DECISION TREE': ['DECISION TREE', 'DT'],
            'GRADIENT BOOSTING': ['GRADIENT BOOSTING', 'XGBoost', 'XGBOOST', 'GBDT'],
            'K-NEAREST NEIGHBORS': ['K-NEAREST NEIGHBOR', 'KNN', 'K-NN'],
            'NAIVE BAYES': ['NAIVE BAYES', 'NB'],
            'ENSEMBLE': ['ENSEMBLE', 'BAGGING', 'BOOSTING'],
            'REGRESSION': ['LINEAR REGRESSION', 'MULTIPLE REGRESSION', 'LOGISTIC REGRESSION'],
            'CLUSTERING': ['K-MEANS', 'CLUSTERING', 'HIERARCHICAL CLUSTERING'],
            'GENETIC ALGORITHM': ['GENETIC ALGORITHM', 'GA'],
            'PARTICLE SWARM': ['PARTICLE SWARM', 'PSO'],
            'RECURRENT NEURAL NETWORK': ['RECURRENT NEURAL NETWORK', 'RNN', 'LSTM', 'GRU'],
            'CONVOLUTIONAL NEURAL NETWORK': ['CONVOLUTIONAL NEURAL NETWORK', 'CNN', 'CONVNET']
        }
        
        for technique, patterns in ai_patterns.items():
            for pattern in patterns:
                if pattern in text_upper:
                    techniques.append(technique)
                    break
        
        return list(set(techniques))
    
    def identify_data_types(self, text):
        """Identify data types mentioned in text"""
        if pd.isna(text):
            return []
        
        text_upper = text.upper()
        data_types = []
        
        data_patterns = {
            'SATELLITE IMAGERY': ['SATELLITE', 'LANDSAT', 'SENTINEL', 'MODIS', 'SPOT'],
            'LIDAR': ['LIDAR', 'LiDAR', 'LASER SCANNING', 'ALS'],
            'DRONE/UAV': ['DRONE', 'UAV', 'UAS', 'UNMANNED AERIAL', 'RPAS'],
            'HYPERSPECTRAL': ['HYPERSPECTRAL', 'HYPERSPEC'],
            'MULTISPECTRAL': ['MULTISPECTRAL', 'MULTISPEC'],
            'RADAR': ['RADAR', 'SAR', 'SYNTHETIC APERTURE'],
            'OPTICAL': ['OPTICAL', 'VISIBLE', 'NIR', 'SWIR'],
            'THERMAL': ['THERMAL', 'TIR', 'TEMPERATURE'],
            'FIELD MEASUREMENTS': ['FIELD MEASUREMENT', 'GROUND TRUTH', 'IN-SITU', 'FIELD DATA'],
            'SPECTRORADIOMETER': ['SPECTRORADIOMETER', 'SPECTROMETER'],
            'PHOTOGRAMMETRY': ['PHOTOGRAMMETRY', 'SfM', 'STRUCTURE FROM MOTION'],
            'METEOROLOGICAL': ['METEOROLOGICAL', 'WEATHER', 'CLIMATE DATA'],
            'TOPOGRAPHIC': ['TOPOGRAPHIC', 'DEM', 'DIGITAL ELEVATION', 'DTM']
        }
        
        for data_type, patterns in data_patterns.items():
            for pattern in patterns:
                if pattern in text_upper:
                    data_types.append(data_type)
                    break
        
        return list(set(data_types))
    
    def analyze_ai_techniques(self):
        """Analyze AI techniques usage"""
        print("Analyzing AI techniques...")
        
        # Combine title, abstract, and keywords for analysis
        self.df_filtered['combined_text'] = (
            self.df_filtered['TI_clean'] + ' ' + 
            self.df_filtered['AB_clean'] + ' ' + 
            self.df_filtered['KW_clean']
        )
        
        # Extract AI techniques
        all_techniques = []
        technique_by_year = {}
        
        for idx, row in self.df_filtered.iterrows():
            techniques = self.identify_ai_techniques(row['combined_text'])
            all_techniques.extend(techniques)
            
            year = row['PY']
            if year not in technique_by_year:
                technique_by_year[year] = []
            technique_by_year[year].extend(techniques)
        
        # Overall technique frequency
        technique_counts = Counter(all_techniques)
        ai_techniques_df = pd.DataFrame(technique_counts.most_common(15), 
                                      columns=['Technique', 'Frequency'])
        
        # Technique trends over time
        technique_trends = {}
        for year in range(2015, 2026):
            year_techniques = Counter(technique_by_year.get(year, []))
            for technique in technique_counts.keys():
                if technique not in technique_trends:
                    technique_trends[technique] = []
                technique_trends[technique].append(year_techniques.get(technique, 0))
        
        trends_df = pd.DataFrame(technique_trends)
        trends_df['Year'] = range(2015, 2026)
        
        return ai_techniques_df, trends_df
    
    def analyze_data_types(self):
        """Analyze data types usage"""
        print("Analyzing data types...")
        
        # Extract data types
        all_data_types = []
        data_type_by_year = {}
        
        for idx, row in self.df_filtered.iterrows():
            data_types = self.identify_data_types(row['combined_text'])
            all_data_types.extend(data_types)
            
            year = row['PY']
            if year not in data_type_by_year:
                data_type_by_year[year] = []
            data_type_by_year[year].extend(data_types)
        
        # Overall data type frequency
        data_type_counts = Counter(all_data_types)
        data_types_df = pd.DataFrame(data_type_counts.most_common(15), 
                                   columns=['Data_Type', 'Frequency'])
        
        # Data type trends over time
        data_type_trends = {}
        for year in range(2015, 2026):
            year_data_types = Counter(data_type_by_year.get(year, []))
            for data_type in data_type_counts.keys():
                if data_type not in data_type_trends:
                    data_type_trends[data_type] = []
                data_type_trends[data_type].append(year_data_types.get(data_type, 0))
        
        trends_df = pd.DataFrame(data_type_trends)
        trends_df['Year'] = range(2015, 2026)
        
        return data_types_df, trends_df
    
    def analyze_drone_usage(self):
        """Analyze drone/UAV usage"""
        print("Analyzing drone/UAV usage...")
        
        # Identify studies using drones
        drone_keywords = ['DRONE', 'UAV', 'UAS', 'UNMANNED AERIAL', 'RPAS']
        
        self.df_filtered['uses_drone'] = self.df_filtered['combined_text'].str.contains(
            '|'.join(drone_keywords), case=False, na=False
        )
        
        drone_studies = self.df_filtered[self.df_filtered['uses_drone']].copy()
        
        # Drone usage by year
        drone_by_year = drone_studies['PY'].value_counts().sort_index()
        
        # Most promising data for drone use
        drone_data_types = []
        for text in drone_studies['combined_text']:
            drone_data_types.extend(self.identify_data_types(text))
        
        drone_data_counts = Counter(drone_data_types)
        promising_drone_data = pd.DataFrame(drone_data_counts.most_common(10), 
                                          columns=['Data_Type', 'Frequency'])
        
        return len(drone_studies), drone_by_year, promising_drone_data
    
    def create_visualizations(self):
        """Create all visualizations"""
        print("Creating visualizations...")
        
        # Load and clean data
        self.load_and_clean_data()
        
        # Perform analyses
        top_authors, top_brazil_authors = self.analyze_authors()
        top_journals, top_brazil_journals = self.analyze_journals()
        top_keywords, top_brazil_keywords = self.analyze_keywords()
        temporal_df = self.analyze_temporal_evolution()
        geographic_df = self.analyze_geographic_distribution()
        ai_techniques_df, ai_trends_df = self.analyze_ai_techniques()
        data_types_df, data_trends_df = self.analyze_data_types()
        drone_count, drone_by_year, promising_drone_data = self.analyze_drone_usage()
        
        # Store results for dashboard
        self.results = {
            'top_authors': top_authors,
            'top_brazil_authors': top_brazil_authors,
            'top_journals': top_journals,
            'top_brazil_journals': top_brazil_journals,
            'top_keywords': top_keywords,
            'top_brazil_keywords': top_brazil_keywords,
            'temporal_df': temporal_df,
            'geographic_df': geographic_df,
            'ai_techniques_df': ai_techniques_df,
            'ai_trends_df': ai_trends_df,
            'data_types_df': data_types_df,
            'data_trends_df': data_trends_df,
            'drone_count': drone_count,
            'drone_by_year': drone_by_year,
            'promising_drone_data': promising_drone_data
        }
        
        return self.results

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BibliometricAnalyzer('/home/ubuntu/Uploads/DF_COMBINADO.csv')
    
    # Run analysis
    results = analyzer.create_visualizations()
    
    print("\n" + "="*50)
    print("BIBLIOMETRIC ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\nDataset Overview:")
    print(f"- Total studies (2015-2025): {len(analyzer.df_filtered)}")
    print(f"- Brazilian studies: {len(analyzer.brazil_df)}")
    print(f"- International studies: {len(analyzer.international_df)}")
    print(f"- Studies using drones/UAV: {results['drone_count']}")
    
    print(f"\nTop 10 Authors:")
    print(results['top_authors'].head(10).to_string(index=False))
    
    print(f"\nTop 10 Journals:")
    print(results['top_journals'].head(10).to_string(index=False))
    
    print(f"\nTop 15 Keywords:")
    print(results['top_keywords'].head(15).to_string(index=False))
    
    print(f"\nTop AI Techniques:")
    print(results['ai_techniques_df'].to_string(index=False))
    
    print(f"\nTop Data Types:")
    print(results['data_types_df'].to_string(index=False))
    
    print(f"\nMost Promising Drone Data:")
    print(results['promising_drone_data'].to_string(index=False))
    
    print(f"\nGeographic Distribution (Top 10):")
    print(results['geographic_df'].head(10).to_string(index=False))
