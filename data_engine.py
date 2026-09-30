"""
data_engine.py - Data loading, cleaning, indexing, and SQLite storage for RiceTec Sales & Opportunity dataset.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import difflib
from typing import Dict, Any, List, Optional, Tuple

EXCEL_FILE = "RiceTec Sales  Opportunity Navpreet.xlsx"
SHEET_NAME = "RiceTec Sales & Opportunity Nav"

class SalesDataEngine:
    def __init__(self, excel_path: str = EXCEL_FILE):
        self.excel_path = excel_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.customer_list: List[str] = []
        self.customer_id_map: Dict[str, str] = {}
        self.customer_name_clean_map: Dict[str, str] = {}
        self.db_conn: Optional[sqlite3.Connection] = None
        self.kpis: Dict[str, Any] = {}
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Loads and cleans the dataset, initializes SQLite database and search indexes."""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Dataset file not found at {self.excel_path}")

        # Load raw sheet
        self.df = pd.read_excel(self.excel_path, sheet_name=SHEET_NAME)
        df = self.df.copy()

        # Clean string columns
        df['Customer_Group__c'] = df['Customer_Group__c'].astype(str).str.strip()
        df['CountyState'] = df['CountyState'].astype(str).str.strip()
        df['District_Name__c'] = df['District_Name__c'].astype(str).str.strip().str.upper()
        df['Region_Name__c'] = df['Region_Name__c'].astype(str).str.strip().str.upper()

        # Split County and State
        def extract_county_state(cs_str: str) -> Tuple[str, str]:
            if ',' in cs_str:
                parts = cs_str.rsplit(',', 1)
                return parts[0].strip(), parts[1].strip().upper()
            return cs_str.strip(), ""

        cs_extracted = df['CountyState'].apply(extract_county_state)
        df['County'] = [c for c, s in cs_extracted]
        df['State'] = [s for c, s in cs_extracted]

        # Extract Customer Name and Customer ID
        def extract_cust_parts(cg: str) -> Tuple[str, str]:
            if ' - ' in cg:
                parts = cg.rsplit(' - ', 1)
                return parts[0].strip(), parts[1].strip()
            return cg.strip(), ""

        cust_extracted = df['Customer_Group__c'].apply(extract_cust_parts)
        df['Customer_Name'] = [name for name, cid in cust_extracted]
        df['Customer_ID'] = [cid for name, cid in cust_extracted]

        # Clean YoY Growth anomaly values (e.g. division by ~0 float artifacts > 10000%)
        # Note: 1.0 = 100% growth. Any growth > 1000 (100,000%) is usually a 0 baseline artifact.
        for yoy_col in ['YoY_Growth_2023', 'YoY_Growth_2024', 'YoY_Growth_2025']:
            if yoy_col in df.columns:
                # Replace inf and astronomical outliers with NaN
                mask = df[yoy_col].abs() > 1000
                df.loc[mask, yoy_col] = np.nan

        # Clean numeric columns (fill NaNs with 0.0 for reliable summation and filtering)
        num_cols = [
            '2023 Acres', '2024 Acres', '2025 Acres',
            'Max_Acres_Per_Customer', 'Max_Yearly_Acres_Per_Customer',
            'Maxp', 'Max_Acres', 'Oppor',
            '2025 Full Page', '2025 Max-Ace', '2025 Non-HT', '2025 Non-HT MG'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Clean rebate status columns (Standardize to 'Yes', 'No', and 'Blank')
        rebate_cols = ['Loyalty Rebate Status', 'Volume Rebate Status', 'Super Loyalty']
        for col in rebate_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.capitalize()
                df[col] = df[col].replace({'Nan': 'Blank', 'None': 'Blank', '': 'Blank', 'False': 'No', 'True': 'Yes'})

        # Derive Opportunity Gap (Oppor - 2025 Acres, if 2025 is less than Oppor)
        df['Opportunity_Gap'] = (df['Oppor'].fillna(0) - df['2025 Acres'].fillna(0)).clip(lower=0)

        self.clean_df = df

        # Build customer search index
        self.customer_list = sorted(list(self.clean_df['Customer_Group__c'].unique()))
        for cg in self.customer_list:
            name, cid = extract_cust_parts(cg)
            if cid:
                self.customer_id_map[cid] = cg
            self.customer_name_clean_map[name.lower()] = cg

        # Initialize SQLite in-memory database
        self._setup_sqlite()

        # Compute pre-calculated KPIs
        self._calculate_kpis()

    def _setup_sqlite(self):
        """Sets up in-memory SQLite database for instant SQL querying."""
        self.db_conn = sqlite3.connect(":memory:", check_same_thread=False)
        # Clean column names for SQL safety (replace spaces and dashes with underscores)
        sql_df = self.clean_df.copy()
        clean_col_names = {c: c.replace(' ', '_').replace('-', '_') for c in sql_df.columns}
        sql_df.rename(columns=clean_col_names, inplace=True)
        sql_df.to_sql("sales_opportunity", self.db_conn, index=False, if_exists="replace")

        # Create indexes
        cursor = self.db_conn.cursor()
        cursor.execute("CREATE INDEX idx_cust ON sales_opportunity (Customer_Group__c);")
        cursor.execute("CREATE INDEX idx_region ON sales_opportunity (Region_Name__c);")
        cursor.execute("CREATE INDEX idx_district ON sales_opportunity (District_Name__c);")
        cursor.execute("CREATE INDEX idx_state ON sales_opportunity (State);")
        cursor.execute("CREATE INDEX idx_county ON sales_opportunity (County);")
        self.db_conn.commit()

    def _calculate_kpis(self):
        """Calculates executive KPI summary metrics."""
        df = self.clean_df
        self.kpis = {
            "total_records": len(df),
            "unique_customers": df['Customer_Group__c'].nunique(),
            "total_2023_acres": float(df['2023 Acres'].sum()),
            "total_2024_acres": float(df['2024 Acres'].sum()),
            "total_2025_acres": float(df['2025 Acres'].sum()),
            "total_oppor": float(df['Oppor'].sum()),
            "total_full_page_2025": float(df['2025 Full Page'].sum()),
            "total_max_ace_2025": float(df['2025 Max-Ace'].sum()),
            "total_non_ht_2025": float(df['2025 Non-HT'].sum()),
            "total_non_ht_mg_2025": float(df['2025 Non-HT MG'].sum()),
            "super_loyalty_count": int((df['Super Loyalty'] == 'Yes').sum()),
            "loyalty_rebate_count": int((df['Loyalty Rebate Status'] == 'Yes').sum()),
            "volume_rebate_count": int((df['Volume Rebate Status'] == 'Yes').sum()),
            "total_regions": int(df['Region_Name__c'].nunique()),
            "total_districts": int(df['District_Name__c'].nunique()),
            "total_states": int(df['State'].nunique()),
            "total_counties": int(df['County'].nunique()),
        }

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Executes a SQL query against the in-memory sales_opportunity table."""
        try:
            return pd.read_sql_query(query, self.db_conn)
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def find_customer_in_query(self, query_str: str) -> Optional[str]:
        """
        Extracts a specific customer from a natural language query sentence.
        Checks for:
        1. 6-10 digit customer ID.
        2. Exact full customer name match.
        3. Token-level matching (e.g. 'DONNY' matches 'DONNY DELINE - 1000006621').
        4. Fuzzy matching on candidate words or short queries.
        """
        import re
        q_clean = query_str.strip().lower()

        # Stop words to ignore when scanning query for customer names
        STOPWORDS = {
            'give', 'show', 'tell', 'what', 'which', 'where', 'when', 'who', 'how', 'much', 'many',
            'seed', 'technology', 'technologies', 'breakdown', 'portfolio', 'mix', 'acres', 'acre', 'crop',
            'status', 'rebate', 'rebates', 'loyalty', 'super', 'volume', 'oppor', 'opportunity', 'opportunities', 'potential',
            'gap', 'headroom', 'growth', 'yoy', 'total', 'average', 'avg', 'sum', 'compare', 'comparison',
            'for', 'the', 'and', 'with', 'from', 'all', 'any', 'each', 'per', 'in', 'on', 'at', 'to', 'of',
            'me', 'us', '2023', '2024', '2025', 'district', 'districts', 'region', 'regions', 'state', 'states',
            'county', 'counties', 'customer', 'customers', 'account', 'accounts', 'farmer', 'farmers', 'grower', 'growers',
            'full', 'page', 'non', 'ht', 'mg', 'max', 'ace', 'details', 'profile', 'data', 'numbers', 'info',
            'information', 'product', 'products', 'category', 'categories', 'value', 'values', 'column', 'columns',
            'top', 'bottom', 'highest', 'lowest', 'biggest', 'smallest', 'rank', 'ranking', 'leaderboard',
            'summary', 'overview', 'executive', 'macro', 'kpi', 'kpis', 'list', 'find',
            'share', 'market', 'strategy', 'strategic', 'recommendations', 'recommendation', 'plan', 'plans',
            'risk', 'risks', 'ricetec', 'increase', 'increasing', 'decrease', 'decreasing', 'analysis', 'insights',
            'explain', 'why', 'factors', 'factor', 'swot', 'outlook', 'forecast', 'target', 'targets',
            'draft', 'pitch', 'adopt', 'adoption', 'write', 'email', 'letter', 'suggest', 'propose',
            'can', 'should', 'would', 'will', 'could', 'may', 'might', 'must', 'help',
            'yes', 'no', 'true', 'false', 'none', 'not', 'but', 'both', 'either', 'neither',
            'grew', 'grow', 'growing', 'grown', 'increased', 'decreased', 'preceding', 'following',
            'compared', 'comparing', 'than', 'more', 'less', 'equal', 'greater', 'above', 'below',
            'have', 'has', 'had', 'having', 'without', 'between', 'only', 'qualify', 'qualifying',
            'qualifies', 'qualified', 'eligible', 'eligibility', 'did', 'does', 'do', 'are', 'is',
            'was', 'were', 'been', 'being', 'year', 'years', 'yearly', 'annual', 'annually',
            'same', 'different', 'other', 'another', 'first', 'second', 'third', 'last', 'prior',
            'past', 'next', 'previous'
        }

        # 1. Check for 6+ digit ID
        digits = re.findall(r'\b\d{6,10}\b', query_str)
        for d in digits:
            if d in self.customer_id_map:
                return self.customer_id_map[d]

        # 2. Check if full customer name is in query
        generic_words = {
            'farms', 'farm', 'enterprises', 'llc', 'inc', 'co', 'growers', 'group', 'planting',
            'partnership', 'partners', 'land', 'agri', 'agriculture', 'corp', 'company', 'ltd',
            'plantation', 'operations', 'holdings'
        }
        for name, cg in sorted(self.customer_name_clean_map.items(), key=lambda x: len(x[0]), reverse=True):
            if len(name) >= 4 and name not in generic_words:
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, q_clean):
                    return cg

        # Guard: If query is an aggregation, count, or filter query, do NOT match single candidate tokens to customer names!
        is_query_aggregation = any(term in q_clean for term in [
            'how many', 'how much', 'count', 'which customers', 'which accounts', 'list customers',
            'list accounts', 'filter', 'where', '=', '<', '>', 'grew in', 'grew both', 'compared with',
            'preceding year', 'both 2024 and 2025'
        ])
        if is_query_aggregation:
            return None

        # 3. Extract candidate words from query (excluding stop words)
        words = re.findall(r'[a-zA-Z]{3,}', q_clean)
        candidate_words = [w for w in words if w not in STOPWORDS]
        if not candidate_words:
            return None

        # Check candidate words against tokens of customer names (e.g. 'donny' -> 'DONNY DELINE')
        for w in candidate_words:
            if len(w) >= 4 and w not in generic_words:
                for cg in self.customer_list:
                    c_name = cg.split(' - ')[0].lower()
                    tokens = re.findall(r'[a-zA-Z]+', c_name)
                    if w in tokens:
                        return cg

        # 4. Fuzzy match candidate words against customer tokens (strict cutoff=0.88)
        all_tokens = {}
        for cg in self.customer_list:
            c_name = cg.split(' - ')[0].lower()
            for tok in re.findall(r'[a-zA-Z]{4,}', c_name):
                if tok not in STOPWORDS and tok not in generic_words:
                    all_tokens[tok] = cg

        for w in candidate_words:
            if len(w) >= 4:
                close = difflib.get_close_matches(w, list(all_tokens.keys()), n=1, cutoff=0.88)
                if close:
                    return all_tokens[close[0]]

        # 5. Fallback fuzzy match if query itself is short (e.g. just typing 'donny delin')
        if len(candidate_words) <= 2 and not is_query_aggregation:
            close = difflib.get_close_matches(' '.join(candidate_words), list(self.customer_name_clean_map.keys()), n=1, cutoff=0.75)
            if close:
                return self.customer_name_clean_map[close[0]]

        return None

    def fuzzy_match_customer(self, query_str: str, threshold: float = 0.5) -> List[str]:
        """Finds matching customer groups using exact, ID, substring, or fuzzy matching."""
        # First use sentence finder
        found = self.find_customer_in_query(query_str)
        if found:
            return [found]

        query_clean = query_str.strip().lower()

        # 1. Check if ID matches directly
        if query_clean in self.customer_id_map:
            return [self.customer_id_map[query_clean]]

        # Check if digits in query match any customer ID
        digit_match = ''.join([c for c in query_clean if c.isdigit()])
        if len(digit_match) >= 6 and digit_match in self.customer_id_map:
            return [self.customer_id_map[digit_match]]

        # 2. Check exact customer name match
        if query_clean in self.customer_name_clean_map:
            return [self.customer_name_clean_map[query_clean]]

        # 3. Substring match
        substring_matches = [
            cg for cg in self.customer_list
            if query_clean in cg.lower()
        ]
        if substring_matches:
            return substring_matches[:5]

        # 4. Fuzzy match using difflib
        names_only = list(self.customer_name_clean_map.keys())
        close_names = difflib.get_close_matches(query_clean, names_only, n=5, cutoff=threshold)
        if close_names:
            return [self.customer_name_clean_map[name] for name in close_names]

        return []

    def get_customer_profile(self, customer_group: str) -> Dict[str, Any]:
        """Returns comprehensive aggregated profile for a customer (including multi-county breakdown)."""
        cust_df = self.clean_df[self.clean_df['Customer_Group__c'] == customer_group]
        if cust_df.empty:
            return {}

        total_2023 = cust_df['2023 Acres'].sum()
        total_2024 = cust_df['2024 Acres'].sum()
        total_2025 = cust_df['2025 Acres'].sum()
        total_oppor = cust_df['Oppor'].sum()
        max_acres = cust_df['Max_Acres'].max()

        # Growth calculation
        growth_24 = ((total_2024 - total_2023) / total_2023) if total_2023 and total_2023 > 0 else None
        growth_25 = ((total_2025 - total_2024) / total_2024) if total_2024 and total_2024 > 0 else None

        locations = []
        for _, row in cust_df.iterrows():
            locations.append({
                "county_state": row['CountyState'],
                "district": row['District_Name__c'],
                "region": row['Region_Name__c'],
                "2023_acres": row['2023 Acres'],
                "2024_acres": row['2024 Acres'],
                "2025_acres": row['2025 Acres'],
                "full_page": row['2025 Full Page'],
                "max_ace": row['2025 Max-Ace'],
                "non_ht": row['2025 Non-HT'],
                "non_ht_mg": row['2025 Non-HT MG'],
                "oppor": row['Oppor'],
            })

        rebates = {
            "loyalty_rebate": "Yes" if (cust_df['Loyalty Rebate Status'] == 'Yes').any() else "No",
            "volume_rebate": "Yes" if (cust_df['Volume Rebate Status'] == 'Yes').any() else "No",
            "super_loyalty": "Yes" if (cust_df['Super Loyalty'] == 'Yes').any() else "No",
        }

        tech_breakdown = {
            "2025 Full Page": float(cust_df['2025 Full Page'].sum()),
            "2025 Max-Ace": float(cust_df['2025 Max-Ace'].sum()),
            "2025 Non-HT": float(cust_df['2025 Non-HT'].sum()),
            "2025 Non-HT MG": float(cust_df['2025 Non-HT MG'].sum()),
        }

        return {
            "customer_group": customer_group,
            "row_count": len(cust_df),
            "total_2023_acres": round(float(total_2023), 2),
            "total_2024_acres": round(float(total_2024), 2),
            "total_2025_acres": round(float(total_2025), 2),
            "growth_2024": round(float(growth_24 * 100), 2) if growth_24 is not None else "N/A",
            "growth_2025": round(float(growth_25 * 100), 2) if growth_25 is not None else "N/A",
            "total_oppor": round(float(total_oppor), 2),
            "max_acres": round(float(max_acres), 2) if pd.notna(max_acres) else "N/A",
            "rebates": rebates,
            "tech_breakdown": tech_breakdown,
            "locations": locations
        }
