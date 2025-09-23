import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PatternMiner:
    """Handles market basket analysis and association rule mining"""
    
    def __init__(self):
        self.frequent_itemsets = None
        self.association_rules = None
        self.transactions = None
        
    def prepare_transactions(self, df, min_transactions_per_user=2):
        """Prepare transaction data for market basket analysis"""
        # Filter for borrow actions only
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0:
            return None
        
        # Group by user to create transaction baskets
        user_books = borrow_df.groupby('user_id')['title'].apply(list).reset_index()
        user_books.columns = ['user_id', 'books']
        
        # Filter users with minimum number of transactions
        user_books['num_books'] = user_books['books'].apply(len)
        user_books = user_books[user_books['num_books'] >= min_transactions_per_user]
        
        if len(user_books) == 0:
            return None
        
        # Extract transactions as list of lists
        transactions = user_books['books'].tolist()
        
        self.transactions = transactions
        return transactions
    
    def create_basket_matrix(self, transactions):
        """Create a binary matrix for market basket analysis"""
        if not transactions:
            return None
        
        # Use TransactionEncoder to create binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_basket
    
    def find_frequent_itemsets(self, df_basket, min_support=0.05, method='apriori'):
        """Find frequent itemsets using Apriori or FP-Growth"""
        if df_basket is None or len(df_basket) == 0:
            return None
        
        try:
            if method.lower() == 'fpgrowth':
                frequent_itemsets = fpgrowth(df_basket, min_support=min_support, use_colnames=True)
            else:
                frequent_itemsets = apriori(df_basket, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                return None
                
            # Sort by support
            frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
            
            self.frequent_itemsets = frequent_itemsets
            return frequent_itemsets
            
        except Exception as e:
            print(f"Error finding frequent itemsets: {str(e)}")
            return None
    
    def generate_association_rules(self, df, min_support=0.05, min_confidence=0.5, min_lift=1.0, method='apriori'):
        """Generate association rules from the dataset"""
        try:
            # Prepare transactions
            transactions = self.prepare_transactions(df)
            if not transactions:
                return pd.DataFrame()
            
            # Create basket matrix
            df_basket = self.create_basket_matrix(transactions)
            if df_basket is None:
                return pd.DataFrame()
            
            # Find frequent itemsets
            frequent_itemsets = self.find_frequent_itemsets(df_basket, min_support, method)
            if frequent_itemsets is None or len(frequent_itemsets) == 0:
                return pd.DataFrame()
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )
            
            if len(rules) == 0:
                return pd.DataFrame()
            
            # Filter by lift
            rules = rules[rules['lift'] >= min_lift]
            
            if len(rules) == 0:
                return pd.DataFrame()
            
            # Sort by lift descending
            rules = rules.sort_values('lift', ascending=False)
            
            # Add rule strength categories
            rules['rule_strength'] = pd.cut(
                rules['lift'],
                bins=[0, 1.2, 2.0, float('inf')],
                labels=['Weak', 'Moderate', 'Strong'],
                include_lowest=True
            )
            
            self.association_rules = rules
            return rules
            
        except Exception as e:
            print(f"Error generating association rules: {str(e)}")
            return pd.DataFrame()
    
    def get_book_recommendations(self, book_title, rules_df, top_n=5):
        """Get book recommendations based on association rules"""
        if rules_df is None or len(rules_df) == 0:
            return []
        
        recommendations = []
        
        # Find rules where the book is in antecedents
        for idx, rule in rules_df.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            if book_title in antecedents:
                for book in consequents:
                    if book != book_title:
                        recommendations.append({
                            'recommended_book': book,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        # Sort by lift and confidence
        recommendations = sorted(
            recommendations,
            key=lambda x: (x['lift'], x['confidence']),
            reverse=True
        )
        
        return recommendations[:top_n]
    
    def create_network_visualization(self, rules_df, max_rules=20):
        """Create network visualization of association rules"""
        if rules_df is None or len(rules_df) == 0:
            return None
        
        try:
            # Limit number of rules for better visualization
            rules_subset = rules_df.head(max_rules)
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for idx, rule in rules_subset.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # Add nodes
                for item in antecedents + consequents:
                    if not G.has_node(item):
                        G.add_node(item)
                
                # Add edges
                for ant in antecedents:
                    for con in consequents:
                        G.add_edge(
                            ant, con,
                            weight=rule['lift'],
                            confidence=rule['confidence'],
                            support=rule['support']
                        )
            
            # Create Pyvis network
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            
            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Generate HTML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                net.save_graph(f.name)
                with open(f.name, 'r') as html_file:
                    html_string = html_file.read()
                os.unlink(f.name)
            
            return html_string
            
        except Exception as e:
            print(f"Error creating network visualization: {str(e)}")
            return None
    
    def get_frequent_patterns_summary(self):
        """Get summary statistics of frequent patterns"""
        if self.frequent_itemsets is None:
            return None
        
        summary = {
            'total_itemsets': len(self.frequent_itemsets),
            'avg_support': self.frequent_itemsets['support'].mean(),
            'max_support': self.frequent_itemsets['support'].max(),
            'min_support': self.frequent_itemsets['support'].min(),
            'itemset_sizes': self.frequent_itemsets['itemsets'].apply(len).value_counts().to_dict()
        }
        
        return summary
    
    def get_rules_summary(self):
        """Get summary statistics of association rules"""
        if self.association_rules is None:
            return None
        
        summary = {
            'total_rules': len(self.association_rules),
            'avg_confidence': self.association_rules['confidence'].mean(),
            'avg_lift': self.association_rules['lift'].mean(),
            'avg_support': self.association_rules['support'].mean(),
            'strong_rules': len(self.association_rules[self.association_rules['lift'] >= 2.0]),
            'moderate_rules': len(self.association_rules[
                (self.association_rules['lift'] >= 1.2) & 
                (self.association_rules['lift'] < 2.0)
            ]),
            'weak_rules': len(self.association_rules[self.association_rules['lift'] < 1.2])
        }
        
        return summary
    
    def analyze_book_relationships(self, df, book_title):
        """Analyze relationships for a specific book"""
        if self.association_rules is None:
            return None
        
        relationships = {
            'appears_with': [],
            'leads_to': [],
            'led_by': []
        }
        
        for idx, rule in self.association_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Book appears in antecedents
            if book_title in antecedents:
                for book in consequents:
                    relationships['leads_to'].append({
                        'book': book,
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
            
            # Book appears in consequents
            if book_title in consequents:
                for book in antecedents:
                    relationships['led_by'].append({
                        'book': book,
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
        
        return relationships
    
    def find_similar_users(self, df, target_user_id, min_common_books=2):
        """Find users with similar reading patterns"""
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        # Get target user's books
        target_books = set(borrow_df[borrow_df['user_id'] == target_user_id]['title'].tolist())
        
        if len(target_books) == 0:
            return []
        
        # Find similar users
        similar_users = []
        
        for user_id in borrow_df['user_id'].unique():
            if user_id == target_user_id:
                continue
            
            user_books = set(borrow_df[borrow_df['user_id'] == user_id]['title'].tolist())
            common_books = target_books.intersection(user_books)
            
            if len(common_books) >= min_common_books:
                similarity = len(common_books) / len(target_books.union(user_books))
                similar_users.append({
                    'user_id': user_id,
                    'common_books': len(common_books),
                    'similarity_score': similarity,
                    'common_titles': list(common_books)
                })
        
        # Sort by similarity score
        similar_users = sorted(similar_users, key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_users