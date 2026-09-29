"""
query_engine.py - Natural Language Query & Reasoning Engine for RiceTec Sales & Opportunity.
Provides deterministic, zero-hallucination answers, aggregations, customer lookups,
rankings, regional/technology breakdowns, and optional LLM assistance.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from data_engine import SalesDataEngine

# Auto-load private environment variables from .env
load_dotenv()

class QueryEngine:
    def __init__(self, data_engine: Optional[SalesDataEngine] = None, gemini_api_key: Optional[str] = None):
        self.de = data_engine if data_engine else SalesDataEngine()
        self.df = self.de.clean_df
        self.gemini_api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    def process_query(self, user_query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Main query processing entry point.
        Analyzes intent and generates markdown response, structured table, and chart data.
        """
        q = user_query.strip()
        q_lower = q.lower()
        active_api_key = api_key if api_key is not None else self.gemini_api_key

        # 1. Greetings & Help
        if any(q_lower == g for g in ["hi", "hello", "hey", "help", "start", "menu"]):
            return self._handle_help_query()

        # 2. Check if a specific customer is mentioned in the query FIRST
        matched_cust = self.de.find_customer_in_query(q)
        if matched_cust:
            return self._handle_customer_query(matched_cust, q_lower)

        # 3. Top / Bottom Rankings (Top N customers, top districts, top regions, top counties)
        if any(term in q_lower for term in ["top", "highest", "largest", "biggest", "bottom", "lowest", "smallest", "leaderboard", "rank"]):
            return self._handle_ranking_query(q_lower)

        # 4. Region Specific Query (R01, R02, R03, R04 or 'region')
        region_match = re.search(r'\b(r0[1-4]|r[1-4])\b', q_lower)
        if region_match or ("region" in q_lower and "customer" not in q_lower):
            return self._handle_region_query(q_lower, region_match.group(0).upper() if region_match else None)

        # 5. District Specific Query (D01 to D16)
        district_match = re.search(r'\b(d0[1-9]|d1[0-6]|d[1-9])\b', q_lower)
        if district_match or ("district" in q_lower and "customer" not in q_lower):
            return self._handle_district_query(q_lower, district_match.group(0).upper() if district_match else None)

        # 6. State Specific Query (AR, LA, MO, TX, MS, TN, IL, GA, FL or full state names)
        state_code = self._extract_state(q_lower)
        if state_code and "customer" not in q_lower and "top" not in q_lower:
            return self._handle_state_query(q_lower, state_code)

        # 7. Portfolio-wide Product Technology Mix (Full Page, Max-Ace, Non-HT, Non-HT MG)
        if any(term in q_lower for term in ["product mix", "seed tech", "technology", "full page", "max-ace", "non-ht", "non ht"]):
            return self._handle_tech_mix_query(q_lower)

        # 8. Portfolio-wide Rebate & Loyalty Queries (Super Loyalty, Loyalty Rebate, Volume Rebate)
        if any(term in q_lower for term in ["rebate", "loyalty", "super loyalty", "volume rebate"]):
            return self._handle_rebate_query(q_lower)

        # 9. Portfolio-wide Opportunity Gap / Unrealized Potential
        if any(term in q_lower for term in ["opportunity gap", "unrealized", "potential gap", "biggest gap", "headroom"]):
            return self._handle_opportunity_gap_query(q_lower)

        # 10. Dataset Overview / Executive Summary
        if any(term in q_lower for term in ["overview", "summary", "kpis", "executive summary", "dataset overview", "macro"]):
            return self._handle_overview_query()

        # 11. Yearly Aggregations & Comparison (2023, 2024, 2025 Acres & YoY Growth)
        if any(year in q_lower for year in ["2023", "2024", "2025", "total acres", "acres"]) and any(term in q_lower for term in ["total", "sum", "average", "avg", "compare", "growth"]):
            return self._handle_yearly_aggregation_query(q_lower)

        # 12. Fuzzy match fallback for customer
        customer_matches = self.de.fuzzy_match_customer(q)
        if customer_matches:
            return self._handle_customer_query(customer_matches[0], q_lower)

        # 13. AI Reasoning via Gemini if API key is provided or loaded from .env
        if active_api_key:
            return self._handle_llm_query(q, active_api_key)

        # Default fallback: General smart search or SQL translation
        return self._handle_smart_fallback(q)

    def _is_customer_query(self, q_lower: str) -> bool:
        indicators = ["customer", "client", "who is", "tell me about", "profile", "account", "farmer", "grower", "acres for", "details on"]
        return any(ind in q_lower for ind in indicators) or any(c.isdigit() for c in q_lower)

    def _extract_state(self, q_lower: str) -> Optional[str]:
        state_map = {
            "arkansas": "AR", "ar": "AR",
            "louisiana": "LA", "la": "LA",
            "missouri": "MO", "mo": "MO",
            "texas": "TX", "tx": "TX",
            "mississippi": "MS", "ms": "MS",
            "tennessee": "TN", "tn": "TN",
            "illinois": "IL", "il": "IL",
            "georgia": "GA", "ga": "GA",
            "florida": "FL", "fl": "FL"
        }
        words = re.findall(r'\b[a-zA-Z]+\b', q_lower)
        for w in words:
            if w in state_map:
                # Avoid confusing common words with state abbreviations unless uppercase or explicitly in context
                if w == "in" or w == "or":
                    continue
                if len(w) == 2 and w not in ["ar", "la", "mo", "tx", "ms", "tn"]:
                    continue
                return state_map[w]
        return None

    def _handle_help_query(self) -> Dict[str, Any]:
        text = (
            "### 🌾 RiceTec Sales & Opportunity Assistant\n\n"
            "I can answer inquiries on sales acreages, customer profiles, growth trends, technology mix, and rebates.\n\n"
            "**Here are quick questions you can ask me:**\n"
            "- **Customer Profiles**: `Tell me about DONNY DELINE`, `Craig Zaunbrecher`, or search by ID `1000006621`\n"
            "- **Rankings**: `Top 10 customers by 2025 acres`, `Top customers in Louisiana`, `Top 5 districts by opportunity`\n"
            "- **Macro & Regional**: `Acres breakdown by Region`, `Summarize Region R01`, `District D11 overview`\n"
            "- **Seed Technology Mix**: `What is the 2025 product mix?`, `Total Full Page acres`, `Max-Ace acres`\n"
            "- **Rebates & Loyalty**: `How many Super Loyalty customers?`, `Volume rebate customers in AR`\n"
            "- **Opportunity & Growth**: `Who has the biggest opportunity gap?`, `Fastest growing customers in 2025`\n"
            "- **Yearly Trends**: `Compare 2024 vs 2025 total acres`\n"
        )
        return {
            "text": text,
            "table": None,
            "chart": None,
            "metric_cards": [
                {"label": "Total Customers", "value": f"{self.de.kpis['unique_customers']:,}"},
                {"label": "2025 Total Acres", "value": f"{self.de.kpis['total_2025_acres']:,.0f}"},
                {"label": "Total Opportunity", "value": f"{self.de.kpis['total_oppor']:,.0f}"},
                {"label": "Districts", "value": f"{self.de.kpis['total_districts']}"},
            ]
        }

    def _handle_overview_query(self) -> Dict[str, Any]:
        k = self.de.kpis
        text = (
            "### 📊 RiceTec Sales & Opportunity - Executive Overview\n\n"
            f"- **Customer Base**: **{k['unique_customers']:,}** unique customers across **{k['total_records']:,}** land entries in **{k['total_counties']}** counties across **{k['total_states']}** states.\n"
            f"- **2025 Total Acres**: **{k['total_2025_acres']:,.1f}** acres (compared to **{k['total_2024_acres']:,.1f}** in 2024 and **{k['total_2023_acres']:,.1f}** in 2023).\n"
            f"- **Total Opportunity (Oppor)**: **{k['total_oppor']:,.1f}** acres.\n"
            f"- **Super Loyalty Status**: **{k['super_loyalty_count']}** customers qualify for Super Loyalty.\n"
            f"- **Rebate Participation**: **{k['loyalty_rebate_count']}** Loyalty Rebate eligible, **{k['volume_rebate_count']}** Volume Rebate eligible.\n\n"
            "#### 2025 Seed Technology Breakdown:\n"
            f"- **Full Page**: {k['total_full_page_2025']:,.1f} acres ({k['total_full_page_2025']/k['total_2025_acres']*100:.1f}%)\n"
            f"- **Non-HT**: {k['total_non_ht_2025']:,.1f} acres ({k['total_non_ht_2025']/k['total_2025_acres']*100:.1f}%)\n"
            f"- **Max-Ace**: {k['total_max_ace_2025']:,.1f} acres ({k['total_max_ace_2025']/k['total_2025_acres']*100:.1f}%)\n"
            f"- **Non-HT MG**: {k['total_non_ht_mg_2025']:,.1f} acres ({k['total_non_ht_mg_2025']/k['total_2025_acres']*100:.1f}%)\n"
        )
        tech_df = pd.DataFrame([
            {"Technology": "Full Page", "2025 Acres": round(k['total_full_page_2025'], 1), "Share %": round(k['total_full_page_2025']/k['total_2025_acres']*100, 1)},
            {"Technology": "Non-HT", "2025 Acres": round(k['total_non_ht_2025'], 1), "Share %": round(k['total_non_ht_2025']/k['total_2025_acres']*100, 1)},
            {"Technology": "Max-Ace", "2025 Acres": round(k['total_max_ace_2025'], 1), "Share %": round(k['total_max_ace_2025']/k['total_2025_acres']*100, 1)},
            {"Technology": "Non-HT MG", "2025 Acres": round(k['total_non_ht_mg_2025'], 1), "Share %": round(k['total_non_ht_mg_2025']/k['total_2025_acres']*100, 1)},
        ])
        return {
            "text": text,
            "table": tech_df,
            "chart": {
                "type": "bar",
                "df": tech_df,
                "x": "Technology",
                "y": "2025 Acres",
                "title": "2025 Acres by Technology"
            },
            "metric_cards": [
                {"label": "Unique Customers", "value": f"{k['unique_customers']:,}"},
                {"label": "2025 Acres", "value": f"{k['total_2025_acres']:,.0f}"},
                {"label": "Total Opportunity", "value": f"{k['total_oppor']:,.0f}"},
                {"label": "Super Loyalty", "value": f"{k['super_loyalty_count']}"}
            ]
        }

    def _handle_customer_query(self, customer_group: str, q_lower: str) -> Dict[str, Any]:
        profile = self.de.get_customer_profile(customer_group)
        if not profile:
            return {"text": f"Could not find customer record for `{customer_group}`.", "table": None, "chart": None, "metric_cards": None}

        c_name = profile['customer_group']
        t_23 = profile['total_2023_acres']
        t_24 = profile['total_2024_acres']
        t_25 = profile['total_2025_acres']
        oppor = profile['total_oppor']
        max_ac = profile['max_acres']
        g_24 = profile['growth_2024']
        g_25 = profile['growth_2025']
        reb = profile['rebates']
        tech = profile['tech_breakdown']

        g24_str = f"{g_24:+}% YoY" if isinstance(g_24, (int, float)) else "N/A"
        g25_str = f"{g_25:+}% YoY" if isinstance(g_25, (int, float)) else "N/A"

        # Sub-intent 1: Specific request for Customer's 2025 Seed Technology Breakdown
        if any(term in q_lower for term in ["technology", "seed tech", "product mix", "full page", "max-ace", "non-ht", "seed"]):
            fp = tech['2025 Full Page']
            ma = tech['2025 Max-Ace']
            nht = tech['2025 Non-HT']
            nmg = tech['2025 Non-HT MG']
            tot = t_25 if t_25 > 0 else 1.0

            text = (
                f"### 🌾 2025 Seed Technology Breakdown: {c_name}\n\n"
                f"- **Customer**: **`{c_name}`**\n"
                f"- **2025 Total Acres**: **`{t_25:,.1f}` acres**\n\n"
                f"**Columns under 2025 Seed Technology Category:**\n"
                f"- **2025 Full Page**: **`{fp:,.1f}` acres** ({fp/tot*100:.1f}% share)\n"
                f"- **2025 Non-HT**: **`{nht:,.1f}` acres** ({nht/tot*100:.1f}% share)\n"
                f"- **2025 Max-Ace**: **`{ma:,.1f}` acres** ({ma/tot*100:.1f}% share)\n"
                f"- **2025 Non-HT MG**: **`{nmg:,.1f}` acres** ({nmg/tot*100:.1f}% share)\n"
            )
            if profile['row_count'] > 1:
                text += f"\n> 📍 *Note: Land spans across **{profile['row_count']} distinct locations** (detailed in the table below).*"

            tech_summary_df = pd.DataFrame([
                {"2025 Seed Technology Column": "2025 Full Page", "Acres": round(fp, 1), "Share %": f"{fp/tot*100:.1f}%"},
                {"2025 Seed Technology Column": "2025 Non-HT", "Acres": round(nht, 1), "Share %": f"{nht/tot*100:.1f}%"},
                {"2025 Seed Technology Column": "2025 Max-Ace", "Acres": round(ma, 1), "Share %": f"{ma/tot*100:.1f}%"},
                {"2025 Seed Technology Column": "2025 Non-HT MG", "Acres": round(nmg, 1), "Share %": f"{nmg/tot*100:.1f}%"},
            ])
            chart_df = pd.DataFrame([
                {"Technology": "2025 Full Page", "2025 Acres": round(fp, 1)},
                {"Technology": "2025 Non-HT", "2025 Acres": round(nht, 1)},
                {"Technology": "2025 Max-Ace", "2025 Acres": round(ma, 1)},
                {"Technology": "2025 Non-HT MG", "2025 Acres": round(nmg, 1)},
            ])
            chart_df = chart_df[chart_df["2025 Acres"] > 0]
            if chart_df.empty:
                chart_df = pd.DataFrame([{"Technology": "None Recorded", "2025 Acres": 0}])

            return {
                "text": text,
                "table": tech_summary_df,
                "chart": {
                    "type": "bar",
                    "df": chart_df,
                    "x": "Technology",
                    "y": "2025 Acres",
                    "title": f"2025 Seed Technology Mix - {c_name.split(' - ')[0]}"
                },
                "metric_cards": [
                    {"label": "2025 Total Acres", "value": f"{t_25:,.1f}"},
                    {"label": "2025 Full Page", "value": f"{fp:,.1f}"},
                    {"label": "2025 Non-HT", "value": f"{nht:,.1f}"},
                    {"label": "2025 Max-Ace", "value": f"{ma:,.1f}"}
                ]
            }

        # Sub-intent 2: Specific request for Rebate / Loyalty status
        if any(term in q_lower for term in ["rebate", "loyalty", "super loyalty"]):
            text = (
                f"### 🎁 Rebate & Loyalty Status: {c_name}\n\n"
                f"- **Super Loyalty Status**: **{reb['super_loyalty']}**\n"
                f"- **Loyalty Rebate Status**: **{reb['loyalty_rebate']}**\n"
                f"- **Volume Rebate Status**: **{reb['volume_rebate']}**\n\n"
                f"- **2025 Total Acres**: `{t_25:,.1f}` acres\n"
                f"- **Market Opportunity (Oppor)**: `{oppor:,.1f}` acres\n"
            )
            reb_df = pd.DataFrame([
                {"Rebate Program": "Super Loyalty", "Status": reb['super_loyalty']},
                {"Rebate Program": "Loyalty Rebate", "Status": reb['loyalty_rebate']},
                {"Rebate Program": "Volume Rebate", "Status": reb['volume_rebate']},
            ])
            return {
                "text": text,
                "table": reb_df,
                "chart": None,
                "metric_cards": [
                    {"label": "Super Loyalty", "value": reb['super_loyalty']},
                    {"label": "Loyalty Rebate", "value": reb['loyalty_rebate']},
                    {"label": "Volume Rebate", "value": reb['volume_rebate']}
                ]
            }

        # Sub-intent 3: Specific request for Opportunity / Potential
        if any(term in q_lower for term in ["oppor", "opportunity", "potential", "headroom", "gap"]):
            gap = max(0, oppor - t_25)
            realization = (t_25 / oppor * 100) if oppor > 0 else 0
            text = (
                f"### 🎯 Opportunity Analysis: {c_name}\n\n"
                f"- **Total Opportunity (Oppor)**: **`{oppor:,.1f}` acres**\n"
                f"- **2025 Realized Acres**: **`{t_25:,.1f}` acres**\n"
                f"- **Untapped Opportunity Gap**: **`{gap:,.1f}` acres**\n"
                f"- **Opportunity Realization**: **`{realization:.1f}%`**\n"
                f"- **Max Recorded Historical Acres**: `{max_ac}` acres\n"
            )
            gap_df = pd.DataFrame([
                {"Metric": "2025 Realized Acres", "Acres": round(t_25, 1)},
                {"Metric": "Untapped Opportunity Gap", "Acres": round(gap, 1)},
            ])
            return {
                "text": text,
                "table": gap_df,
                "chart": {
                    "type": "bar",
                    "df": gap_df,
                    "x": "Metric",
                    "y": "Acres",
                    "title": f"Opportunity vs Realized - {c_name.split(' - ')[0]}"
                },
                "metric_cards": [
                    {"label": "Market Oppor", "value": f"{oppor:,.1f}"},
                    {"label": "2025 Acres", "value": f"{t_25:,.1f}"},
                    {"label": "Untapped Gap", "value": f"{gap:,.1f}"},
                    {"label": "Realization", "value": f"{realization:.1f}%"}
                ]
            }

        text = (
            f"### 🌾 Customer Profile: {c_name}\n\n"
            f"- **Acreage History**: \n"
            f"  - **2023 Acres**: `{t_23:,.1f}`\n"
            f"  - **2024 Acres**: `{t_24:,.1f}` ({g24_str})\n"
            f"  - **2025 Acres**: `{t_25:,.1f}` ({g25_str})\n"
            f"- **Opportunity & Max Potential**: \n"
            f"  - **Total Opportunity (Oppor)**: `{oppor:,.1f}` acres\n"
            f"  - **Max Acres Recorded**: `{max_ac}` acres\n"
            f"- **Rebate & Loyalty Status**:\n"
            f"  - Super Loyalty: **{reb['super_loyalty']}**\n"
            f"  - Loyalty Rebate: **{reb['loyalty_rebate']}**\n"
            f"  - Volume Rebate: **{reb['volume_rebate']}**\n\n"
            f"#### 2025 Seed Technology Breakdown:\n"
            f"- **Full Page**: `{tech['2025 Full Page']:,.1f}` acres\n"
            f"- **Max-Ace**: `{tech['2025 Max-Ace']:,.1f}` acres\n"
            f"- **Non-HT**: `{tech['2025 Non-HT']:,.1f}` acres\n"
            f"- **Non-HT MG**: `{tech['2025 Non-HT MG']:,.1f}` acres\n"
        )

        if profile['row_count'] > 1:
            text += f"\n> 📍 *Note: This customer holds land across **{profile['row_count']} distinct county/district locations** listed in the table below.*"

        loc_df = pd.DataFrame(profile['locations'])
        loc_df.rename(columns={
            "county_state": "Location (County, State)",
            "district": "District",
            "region": "Region",
            "2023_acres": "2023 Acres",
            "2024_acres": "2024 Acres",
            "2025_acres": "2025 Acres",
            "full_page": "Full Page",
            "max_ace": "Max-Ace",
            "non_ht": "Non-HT",
            "non_ht_mg": "Non-HT MG",
            "oppor": "Oppor"
        }, inplace=True)

        yearly_df = pd.DataFrame([
            {"Year": "2023", "Acres": t_23},
            {"Year": "2024", "Acres": t_24},
            {"Year": "2025", "Acres": t_25}
        ])

        return {
            "text": text,
            "table": loc_df,
            "chart": {
                "type": "bar",
                "df": yearly_df,
                "x": "Year",
                "y": "Acres",
                "title": f"Acreage Trend for {c_name.split(' - ')[0]}"
            },
            "metric_cards": [
                {"label": "2025 Acres", "value": f"{t_25:,.1f}", "delta": f"{g_25}% vs 2024" if g_25 != "N/A" else None},
                {"label": "Oppor (Potential)", "value": f"{oppor:,.1f}"},
                {"label": "Super Loyalty", "value": reb['super_loyalty']},
                {"label": "Locations", "value": f"{profile['row_count']}"}
            ]
        }

    def _handle_ranking_query(self, q_lower: str) -> Dict[str, Any]:
        # Extract N (e.g. top 5, top 10, top 20)
        n_match = re.search(r'\b(top|bottom)\s*(\d+)\b', q_lower)
        n = int(n_match.group(2)) if n_match else 10
        is_bottom = "bottom" in q_lower or "lowest" in q_lower or "smallest" in q_lower
        ascending = is_bottom

        # Filter by state if present
        state = self._extract_state(q_lower)
        df_target = self.df.copy()
        state_filter_msg = ""
        if state:
            df_target = df_target[df_target['State'] == state]
            state_filter_msg = f" in **{state}**"

        # Determine metric to rank on
        if "growth" in q_lower:
            rank_col = "YoY_Growth_2025" if "2025" in q_lower else ("YoY_Growth_2024" if "2024" in q_lower else "YoY_Growth_2025")
            label = f"{rank_col} %"
            df_target = df_target.dropna(subset=[rank_col])
        elif "oppor" in q_lower or "opportunity" in q_lower:
            rank_col = "Oppor"
            label = "Total Opportunity (Oppor)"
        elif "2023" in q_lower:
            rank_col = "2023 Acres"
            label = "2023 Acres"
        elif "2024" in q_lower:
            rank_col = "2024 Acres"
            label = "2024 Acres"
        else:
            rank_col = "2025 Acres"
            label = "2025 Acres"

        # Grouping entity: District, County, Region, or Customer
        if "district" in q_lower:
            agg_df = df_target.groupby("District_Name__c")[rank_col].sum().reset_index()
            agg_df.rename(columns={"District_Name__c": "District", rank_col: label}, inplace=True)
            agg_df = agg_df.sort_values(by=label, ascending=ascending).head(n)
            entity = "Districts"
            x_col = "District"
        elif "county" in q_lower:
            agg_df = df_target.groupby("CountyState")[rank_col].sum().reset_index()
            agg_df.rename(columns={"CountyState": "County / State", rank_col: label}, inplace=True)
            agg_df = agg_df.sort_values(by=label, ascending=ascending).head(n)
            entity = "Counties"
            x_col = "County / State"
        elif "region" in q_lower:
            agg_df = df_target.groupby("Region_Name__c")[rank_col].sum().reset_index()
            agg_df.rename(columns={"Region_Name__c": "Region", rank_col: label}, inplace=True)
            agg_df = agg_df.sort_values(by=label, ascending=ascending).head(n)
            entity = "Regions"
            x_col = "Region"
        else:
            # Customer ranking
            # Aggregate per customer first (since some have multiple county rows)
            agg_df = df_target.groupby(["Customer_Group__c"])[rank_col].sum().reset_index()
            # Also get primary county/state
            cust_info = df_target.groupby("Customer_Group__c")[['CountyState', 'District_Name__c', 'Region_Name__c', 'Super Loyalty']].first().reset_index()
            agg_df = pd.merge(agg_df, cust_info, on="Customer_Group__c")
            agg_df.rename(columns={"Customer_Group__c": "Customer", rank_col: label}, inplace=True)
            agg_df = agg_df.sort_values(by=label, ascending=ascending).head(n)
            entity = "Customers"
            x_col = "Customer"

        direction_word = "Bottom" if is_bottom else "Top"
        text = (
            f"### 🏆 {direction_word} {len(agg_df)} {entity} by {label}{state_filter_msg}\n\n"
            f"Here are the ranked {entity.lower()}:\n"
        )
        for i, row in agg_df.reset_index(drop=True).iterrows():
            val = row[label]
            val_str = f"{val:,.1f}" if isinstance(val, (int, float)) else str(val)
            name = row[x_col]
            text += f"{i+1}. **{name}**: `{val_str}`\n"

        return {
            "text": text,
            "table": agg_df,
            "chart": {
                "type": "bar",
                "df": agg_df.head(10),
                "x": x_col,
                "y": label,
                "title": f"{direction_word} {entity} by {label}"
            },
            "metric_cards": [
                {"label": f"{direction_word} #1 {entity[:-1]}", "value": str(agg_df.iloc[0][x_col]).split(' - ')[0]},
                {"label": f"#1 {label}", "value": f"{agg_df.iloc[0][label]:,.1f}" if isinstance(agg_df.iloc[0][label], (int, float)) else str(agg_df.iloc[0][label])},
                {"label": f"Total for {direction_word} {len(agg_df)}", "value": f"{agg_df[label].sum():,.1f}" if isinstance(agg_df[label].sum(), (int, float)) else "N/A"}
            ]
        }

    def _handle_tech_mix_query(self, q_lower: str) -> Dict[str, Any]:
        k = self.de.kpis
        tot_25 = k['total_2025_acres']
        fp = k['total_full_page_2025']
        nht = k['total_non_ht_2025']
        ma = k['total_max_ace_2025']
        nmg = k['total_non_ht_mg_2025']

        df_tech = pd.DataFrame([
            {"Technology": "Full Page", "2025 Acres": round(fp, 1), "Share %": round(fp / tot_25 * 100, 1)},
            {"Technology": "Non-HT", "2025 Acres": round(nht, 1), "Share %": round(nht / tot_25 * 100, 1)},
            {"Technology": "Max-Ace", "2025 Acres": round(ma, 1), "Share %": round(ma / tot_25 * 100, 1)},
            {"Technology": "Non-HT MG", "2025 Acres": round(nmg, 1), "Share %": round(nmg / tot_25 * 100, 1)},
        ])

        text = (
            "### 🌾 2025 Seed Technology Portfolio Breakdown\n\n"
            f"The total recorded **2025 Acres** across all seed technologies is **{tot_25:,.1f} acres**:\n\n"
            f"1. **Full Page**: **{fp:,.1f} acres** (**{fp/tot_25*100:.1f}%** share) — The leading product line.\n"
            f"2. **Non-HT**: **{nht:,.1f} acres** (**{nht/tot_25*100:.1f}%** share) — Conventional non-herbicide tolerant rice.\n"
            f"3. **Max-Ace**: **{ma:,.1f} acres** (**{ma/tot_25*100:.1f}%** share) — Herbicide tolerant technology.\n"
            f"4. **Non-HT MG**: **{nmg:,.1f} acres** (**{nmg/tot_25*100:.1f}%** share) — Non-HT medium grain variety.\n"
        )
        return {
            "text": text,
            "table": df_tech,
            "chart": {
                "type": "bar",
                "df": df_tech,
                "x": "Technology",
                "y": "2025 Acres",
                "title": "2025 Seed Technology Acreage"
            },
            "metric_cards": [
                {"label": "Full Page Share", "value": f"{fp/tot_25*100:.1f}%"},
                {"label": "Non-HT Share", "value": f"{nht/tot_25*100:.1f}%"},
                {"label": "Max-Ace Share", "value": f"{ma/tot_25*100:.1f}%"},
                {"label": "Non-HT MG Share", "value": f"{nmg/tot_25*100:.1f}%"}
            ]
        }

    def _handle_rebate_query(self, q_lower: str) -> Dict[str, Any]:
        df = self.df
        total_cust = df['Customer_Group__c'].nunique()

        # Customer level aggregations for rebates
        cust_reb = df.groupby('Customer_Group__c').agg({
            'Super Loyalty': lambda x: 'Yes' if (x == 'Yes').any() else 'No',
            'Loyalty Rebate Status': lambda x: 'Yes' if (x == 'Yes').any() else 'No',
            'Volume Rebate Status': lambda x: 'Yes' if (x == 'Yes').any() else 'No',
            '2025 Acres': 'sum',
            'Oppor': 'sum',
            'Region_Name__c': 'first',
            'District_Name__c': 'first'
        }).reset_index()

        super_loyalty_n = (cust_reb['Super Loyalty'] == 'Yes').sum()
        loyalty_n = (cust_reb['Loyalty Rebate Status'] == 'Yes').sum()
        volume_n = (cust_reb['Volume Rebate Status'] == 'Yes').sum()
        both_n = ((cust_reb['Loyalty Rebate Status'] == 'Yes') & (cust_reb['Volume Rebate Status'] == 'Yes')).sum()

        summary_df = pd.DataFrame([
            {"Program": "Super Loyalty", "Qualifying Customers": int(super_loyalty_n), "% of Base": round(super_loyalty_n / total_cust * 100, 1)},
            {"Program": "Loyalty Rebate", "Qualifying Customers": int(loyalty_n), "% of Base": round(loyalty_n / total_cust * 100, 1)},
            {"Program": "Volume Rebate", "Qualifying Customers": int(volume_n), "% of Base": round(volume_n / total_cust * 100, 1)},
            {"Program": "Both Loyalty & Volume Rebate", "Qualifying Customers": int(both_n), "% of Base": round(both_n / total_cust * 100, 1)},
        ])

        # If specific program asked
        if "super loyalty" in q_lower:
            qual_df = cust_reb[cust_reb['Super Loyalty'] == 'Yes'].sort_values('2025 Acres', ascending=False)
            text = (
                f"### ⭐ Super Loyalty Status Analysis\n\n"
                f"- **{super_loyalty_n:,} customers** ({super_loyalty_n/total_cust*100:.1f}%) qualify for **Super Loyalty**.\n"
                f"- Combined 2025 Acres for Super Loyalty customers: **{qual_df['2025 Acres'].sum():,.1f} acres**.\n"
                f"- Top Super Loyalty customers listed below:\n"
            )
            table_show = qual_df[['Customer_Group__c', '2025 Acres', 'Oppor', 'Region_Name__c', 'District_Name__c']].head(15)
            table_show.rename(columns={'Customer_Group__c': 'Customer', 'Region_Name__c': 'Region', 'District_Name__c': 'District'}, inplace=True)
        else:
            text = (
                "### 🎁 Rebate & Loyalty Status Overview\n\n"
                f"Breakdown of customer eligibility across RiceTec rebate programs:\n\n"
                f"- **Super Loyalty**: **{super_loyalty_n:,} customers** ({super_loyalty_n/total_cust*100:.1f}% of base)\n"
                f"- **Loyalty Rebate**: **{loyalty_n:,} customers** ({loyalty_n/total_cust*100:.1f}% of base)\n"
                f"- **Volume Rebate**: **{volume_n:,} customers** ({volume_n/total_cust*100:.1f}% of base)\n"
                f"- **Both Loyalty + Volume**: **{both_n:,} customers** ({both_n/total_cust*100:.1f}% of base)\n"
            )
            table_show = summary_df

        return {
            "text": text,
            "table": table_show,
            "chart": {
                "type": "bar",
                "df": summary_df,
                "x": "Program",
                "y": "Qualifying Customers",
                "title": "Customer Count by Rebate Program"
            },
            "metric_cards": [
                {"label": "Super Loyalty", "value": f"{super_loyalty_n:,}"},
                {"label": "Loyalty Rebate", "value": f"{loyalty_n:,}"},
                {"label": "Volume Rebate", "value": f"{volume_n:,}"},
                {"label": "Both Programs", "value": f"{both_n:,}"}
            ]
        }

    def _handle_opportunity_gap_query(self, q_lower: str) -> Dict[str, Any]:
        df = self.df.copy()
        # Group by customer
        cust_gap = df.groupby('Customer_Group__c').agg({
            'Oppor': 'sum',
            '2025 Acres': 'sum',
            'Max_Acres': 'max',
            'Region_Name__c': 'first',
            'District_Name__c': 'first',
            'CountyState': 'first'
        }).reset_index()

        cust_gap['Unrealized_Opportunity'] = (cust_gap['Oppor'].fillna(0) - cust_gap['2025 Acres'].fillna(0)).clip(lower=0)
        cust_gap['Realization_Pct'] = (cust_gap['2025 Acres'] / cust_gap['Oppor'].replace(0, np.nan) * 100).round(1)

        top_gaps = cust_gap.sort_values('Unrealized_Opportunity', ascending=False).head(10)

        total_gap = cust_gap['Unrealized_Opportunity'].sum()
        text = (
            "### 🎯 Opportunity & Growth Gap Analysis\n\n"
            f"- **Total Market Opportunity (Oppor)**: **{cust_gap['Oppor'].sum():,.1f} acres**\n"
            f"- **2025 Realized Acres**: **{cust_gap['2025 Acres'].sum():,.1f} acres**\n"
            f"- **Unrealized Opportunity Gap**: **{total_gap:,.1f} acres** ({total_gap / cust_gap['Oppor'].sum() * 100:.1f}% untapped)\n\n"
            "**Top 10 Accounts with the Largest Unrealized Opportunity:**\n"
        )
        for i, row in top_gaps.reset_index(drop=True).iterrows():
            c_short = row['Customer_Group__c'].split(' - ')[0]
            text += f"{i+1}. **{c_short}**: `{row['Unrealized_Opportunity']:,.1f}` acres gap (Oppor: {row['Oppor']:,.1f} | 2025: {row['2025 Acres']:,.1f})\n"

        display_df = top_gaps[['Customer_Group__c', 'Oppor', '2025 Acres', 'Unrealized_Opportunity', 'Realization_Pct', 'Region_Name__c']].copy()
        display_df.rename(columns={
            'Customer_Group__c': 'Customer',
            'Oppor': 'Oppor Acres',
            'Unrealized_Opportunity': 'Untapped Gap (Acres)',
            'Realization_Pct': 'Realization %',
            'Region_Name__c': 'Region'
        }, inplace=True)

        return {
            "text": text,
            "table": display_df,
            "chart": {
                "type": "bar",
                "df": display_df,
                "x": "Customer",
                "y": "Untapped Gap (Acres)",
                "title": "Top 10 Customers with Largest Opportunity Gap"
            },
            "metric_cards": [
                {"label": "Total Market Oppor", "value": f"{cust_gap['Oppor'].sum():,.0f}"},
                {"label": "Total Unrealized Gap", "value": f"{total_gap:,.0f}"},
                {"label": "Avg Realization", "value": f"{cust_gap['Realization_Pct'].mean():.1f}%"}
            ]
        }

    def _handle_region_query(self, q_lower: str, region_code: Optional[str]) -> Dict[str, Any]:
        df = self.df
        if region_code:
            # Normalize e.g. R1 -> R01
            if len(region_code) == 2 and region_code.startswith("R"):
                region_code = f"R0{region_code[1]}"

            reg_df = df[df['Region_Name__c'] == region_code]
            if reg_df.empty:
                return {"text": f"No data found for Region `{region_code}`.", "table": None, "chart": None, "metric_cards": None}

            tot_25 = reg_df['2025 Acres'].sum()
            tot_24 = reg_df['2024 Acres'].sum()
            tot_23 = reg_df['2023 Acres'].sum()
            tot_opp = reg_df['Oppor'].sum()
            districts = sorted(reg_df['District_Name__c'].unique())
            cust_count = reg_df['Customer_Group__c'].nunique()

            dist_breakdown = reg_df.groupby('District_Name__c').agg({
                'Customer_Group__c': 'nunique',
                '2024 Acres': 'sum',
                '2025 Acres': 'sum',
                'Oppor': 'sum'
            }).reset_index()
            dist_breakdown.rename(columns={
                'District_Name__c': 'District',
                'Customer_Group__c': 'Customers',
                '2024 Acres': '2024 Acres',
                '2025 Acres': '2025 Acres',
                'Oppor': 'Oppor'
            }, inplace=True)

            text = (
                f"### 📍 Region {region_code} Summary\n\n"
                f"- **Customers**: **{cust_count:,} unique customers**\n"
                f"- **Districts**: {', '.join(districts)}\n"
                f"- **2025 Acres**: **{tot_25:,.1f} acres**\n"
                f"- **2024 Acres**: **{tot_24:,.1f} acres**\n"
                f"- **2023 Acres**: **{tot_23:,.1f} acres**\n"
                f"- **Total Opportunity (Oppor)**: **{tot_opp:,.1f} acres**\n"
            )

            return {
                "text": text,
                "table": dist_breakdown,
                "chart": {
                    "type": "bar",
                    "df": dist_breakdown,
                    "x": "District",
                    "y": "2025 Acres",
                    "title": f"District Breakdown in Region {region_code}"
                },
                "metric_cards": [
                    {"label": f"Region {region_code} 2025", "value": f"{tot_25:,.0f}"},
                    {"label": "Opportunity", "value": f"{tot_opp:,.0f}"},
                    {"label": "Customers", "value": f"{cust_count:,}"},
                    {"label": "Districts", "value": f"{len(districts)}"}
                ]
            }
        else:
            # Regional comparison
            reg_agg = df.groupby('Region_Name__c').agg({
                'Customer_Group__c': 'nunique',
                '2023 Acres': 'sum',
                '2024 Acres': 'sum',
                '2025 Acres': 'sum',
                'Oppor': 'sum'
            }).reset_index()
            reg_agg.rename(columns={
                'Region_Name__c': 'Region',
                'Customer_Group__c': 'Customers',
                '2023 Acres': '2023 Acres',
                '2024 Acres': '2024 Acres',
                '2025 Acres': '2025 Acres',
                'Oppor': 'Opportunity'
            }, inplace=True)

            text = (
                "### 🗺️ Regional Sales & Opportunity Breakdown\n\n"
                "Summary across all 4 sales regions:\n"
            )
            for _, r in reg_agg.iterrows():
                text += f"- **{r['Region']}**: **{r['2025 Acres']:,.1f} acres** in 2025 ({r['Customers']} customers, Oppor: {r['Opportunity']:,.1f})\n"

            return {
                "text": text,
                "table": reg_agg,
                "chart": {
                    "type": "bar",
                    "df": reg_agg,
                    "x": "Region",
                    "y": "2025 Acres",
                    "title": "2025 Acres by Region"
                },
                "metric_cards": [
                    {"label": "Leading Region", "value": reg_agg.sort_values('2025 Acres', ascending=False).iloc[0]['Region']},
                    {"label": "Top Region Acres", "value": f"{reg_agg['2025 Acres'].max():,.0f}"},
                    {"label": "Total Regions", "value": f"{len(reg_agg)}"}
                ]
            }

    def _handle_district_query(self, q_lower: str, district_code: Optional[str]) -> Dict[str, Any]:
        df = self.df
        if district_code:
            # Normalize e.g. D1 -> D01
            if len(district_code) == 2 and district_code.startswith("D"):
                district_code = f"D0{district_code[1]}"

            dist_df = df[df['District_Name__c'] == district_code]
            if dist_df.empty:
                return {"text": f"No records found for District `{district_code}`.", "table": None, "chart": None, "metric_cards": None}

            tot_25 = dist_df['2025 Acres'].sum()
            tot_24 = dist_df['2024 Acres'].sum()
            tot_opp = dist_df['Oppor'].sum()
            region = dist_df['Region_Name__c'].iloc[0]
            cust_n = dist_df['Customer_Group__c'].nunique()
            counties = dist_df['CountyState'].unique()

            top_cust = dist_df.groupby('Customer_Group__c')['2025 Acres'].sum().reset_index().sort_values('2025 Acres', ascending=False).head(5)

            text = (
                f"### 📌 District {district_code} Details (Region {region})\n\n"
                f"- **Customers**: **{cust_n} customers** across {len(counties)} counties/states\n"
                f"- **2025 Acres**: **{tot_25:,.1f} acres**\n"
                f"- **2024 Acres**: **{tot_24:,.1f} acres**\n"
                f"- **Opportunity (Oppor)**: **{tot_opp:,.1f} acres**\n\n"
                f"**Top Customers in District {district_code}:**\n"
            )
            for i, r in top_cust.reset_index(drop=True).iterrows():
                c_name = r['Customer_Group__c'].split(' - ')[0]
                text += f"{i+1}. **{c_name}**: `{r['2025 Acres']:,.1f}` acres\n"

            top_cust.rename(columns={'Customer_Group__c': 'Customer', '2025 Acres': '2025 Acres'}, inplace=True)

            return {
                "text": text,
                "table": top_cust,
                "chart": {
                    "type": "bar",
                    "df": top_cust,
                    "x": "Customer",
                    "y": "2025 Acres",
                    "title": f"Top Customers in District {district_code}"
                },
                "metric_cards": [
                    {"label": f"District {district_code} 2025", "value": f"{tot_25:,.0f}"},
                    {"label": "Opportunity", "value": f"{tot_opp:,.0f}"},
                    {"label": "Customers", "value": f"{cust_n}"}
                ]
            }
        else:
            # District overview
            dist_agg = df.groupby('District_Name__c').agg({
                'Region_Name__c': 'first',
                'Customer_Group__c': 'nunique',
                '2025 Acres': 'sum',
                'Oppor': 'sum'
            }).reset_index()
            dist_agg.rename(columns={
                'District_Name__c': 'District',
                'Region_Name__c': 'Region',
                'Customer_Group__c': 'Customers',
                '2025 Acres': '2025 Acres',
                'Oppor': 'Oppor'
            }, inplace=True)
            dist_agg = dist_agg.sort_values('2025 Acres', ascending=False)

            text = (
                "### 📌 District Acreage Ranking (All 16 Districts)\n\n"
                "Here are the districts ranked by 2025 Acres:\n"
            )
            for i, r in dist_agg.head(8).reset_index(drop=True).iterrows():
                text += f"{i+1}. **{r['District']}** ({r['Region']}): `{r['2025 Acres']:,.1f}` acres ({r['Customers']} customers)\n"

            return {
                "text": text,
                "table": dist_agg,
                "chart": {
                    "type": "bar",
                    "df": dist_agg.head(10),
                    "x": "District",
                    "y": "2025 Acres",
                    "title": "Top 10 Districts by 2025 Acres"
                },
                "metric_cards": [
                    {"label": "Top District", "value": dist_agg.iloc[0]['District']},
                    {"label": "Top District Acres", "value": f"{dist_agg.iloc[0]['2025 Acres']:,.0f}"},
                    {"label": "Total Districts", "value": f"{len(dist_agg)}"}
                ]
            }

    def _handle_state_query(self, q_lower: str, state_code: str) -> Dict[str, Any]:
        st_df = self.df[self.df['State'] == state_code]
        if st_df.empty:
            return {"text": f"No data found for State `{state_code}`.", "table": None, "chart": None, "metric_cards": None}

        tot_25 = st_df['2025 Acres'].sum()
        tot_24 = st_df['2024 Acres'].sum()
        tot_opp = st_df['Oppor'].sum()
        cust_n = st_df['Customer_Group__c'].nunique()
        counties = st_df['County'].nunique()

        county_agg = st_df.groupby('County').agg({
            'Customer_Group__c': 'nunique',
            '2025 Acres': 'sum',
            'Oppor': 'sum'
        }).reset_index().sort_values('2025 Acres', ascending=False)
        county_agg.rename(columns={'Customer_Group__c': 'Customers'}, inplace=True)

        text = (
            f"### 🇺🇸 State Summary: {state_code}\n\n"
            f"- **Customers**: **{cust_n:,} unique customers** across **{counties} counties**\n"
            f"- **2025 Acres**: **{tot_25:,.1f} acres**\n"
            f"- **2024 Acres**: **{tot_24:,.1f} acres**\n"
            f"- **Opportunity (Oppor)**: **{tot_opp:,.1f} acres**\n\n"
            f"**Top Counties in {state_code}:**\n"
        )
        for i, r in county_agg.head(5).reset_index(drop=True).iterrows():
            text += f"{i+1}. **{r['County']}**: `{r['2025 Acres']:,.1f}` acres ({r['Customers']} customers)\n"

        return {
            "text": text,
            "table": county_agg.head(10),
            "chart": {
                "type": "bar",
                "df": county_agg.head(10),
                "x": "County",
                "y": "2025 Acres",
                "title": f"Top Counties in {state_code} (2025 Acres)"
            },
            "metric_cards": [
                {"label": f"{state_code} 2025 Acres", "value": f"{tot_25:,.0f}"},
                {"label": "Opportunity", "value": f"{tot_opp:,.0f}"},
                {"label": "Customers", "value": f"{cust_n:,}"},
                {"label": "Counties", "value": f"{counties}"}
            ]
        }

    def _handle_yearly_aggregation_query(self, q_lower: str) -> Dict[str, Any]:
        k = self.de.kpis
        y23 = k['total_2023_acres']
        y24 = k['total_2024_acres']
        y25 = k['total_2025_acres']
        opp = k['total_oppor']

        g_24 = ((y24 - y23) / y23 * 100) if y23 else 0
        g_25 = ((y25 - y24) / y24 * 100) if y24 else 0

        summary_df = pd.DataFrame([
            {"Year": "2023", "Total Acres": round(y23, 1), "YoY Growth %": "Baseline"},
            {"Year": "2024", "Total Acres": round(y24, 1), "YoY Growth %": f"{g_24:+.2f}%"},
            {"Year": "2025", "Total Acres": round(y25, 1), "YoY Growth %": f"{g_25:+.2f}%"},
        ])

        text = (
            "### 📈 Yearly Acreage Comparison (2023 - 2025)\n\n"
            f"- **2023 Total Acres**: **{y23:,.1f} acres**\n"
            f"- **2024 Total Acres**: **{y24:,.1f} acres** ({g_24:+.2f}% YoY)\n"
            f"- **2025 Total Acres**: **{y25:,.1f} acres** ({g_25:+.2f}% YoY)\n"
            f"- **Total Opportunity Baseline**: **{opp:,.1f} acres**\n"
        )

        return {
            "text": text,
            "table": summary_df,
            "chart": {
                "type": "bar",
                "df": summary_df,
                "x": "Year",
                "y": "Total Acres",
                "title": "Total Acres by Year"
            },
            "metric_cards": [
                {"label": "2023 Acres", "value": f"{y23:,.0f}"},
                {"label": "2024 Acres", "value": f"{y24:,.0f}", "delta": f"{g_24:+.1f}%"},
                {"label": "2025 Acres", "value": f"{y25:,.0f}", "delta": f"{g_25:+.1f}%"}
            ]
        }

    def _handle_smart_fallback(self, query: str) -> Dict[str, Any]:
        """Fallback for unstructured questions or general keyword searches."""
        q_words = [w.strip() for w in query.split() if len(w) > 2]
        # Search in Customer Name, County, State, District
        mask = pd.Series(False, index=self.df.index)
        for w in q_words:
            mask = mask | self.df['Customer_Group__c'].str.contains(w, case=False, na=False)
            mask = mask | self.df['CountyState'].str.contains(w, case=False, na=False)
            mask = mask | self.df['District_Name__c'].str.contains(w, case=False, na=False)

        results = self.df[mask]
        if not results.empty:
            count = len(results)
            cust_n = results['Customer_Group__c'].nunique()
            text = (
                f"### 🔍 Search Results for: *\"{query}\"*\n\n"
                f"Found **{count} matching records** across **{cust_n} unique customers**.\n"
                f"- **2025 Acres**: `{results['2025 Acres'].sum():,.1f}`\n"
                f"- **Opportunity (Oppor)**: `{results['Oppor'].sum():,.1f}`\n"
            )
            preview_df = results[['Customer_Group__c', 'CountyState', '2024 Acres', '2025 Acres', 'Oppor', 'District_Name__c']].head(15)
            preview_df.rename(columns={'Customer_Group__c': 'Customer', 'CountyState': 'Location'}, inplace=True)
            return {
                "text": text,
                "table": preview_df,
                "chart": None,
                "metric_cards": [
                    {"label": "Matches", "value": f"{count}"},
                    {"label": "Unique Customers", "value": f"{cust_n}"},
                    {"label": "Total 2025 Acres", "value": f"{results['2025 Acres'].sum():,.1f}"}
                ]
            }

        return {
            "text": (
                f"I couldn't pinpoint a specific customer, metric, or region for *\"{query}\"*.\n\n"
                "**Try asking:**\n"
                "- A customer name: `Tell me about DONNY DELINE`\n"
                "- A region or district: `Summarize Region R01` or `District D11`\n"
                "- A ranking: `Top 10 customers in 2025`\n"
                "- Product mix: `2025 Full Page vs Max-Ace`\n"
                "- Rebates: `Super Loyalty customers`\n"
            ),
            "table": None,
            "chart": None,
            "metric_cards": None
        }

    def _handle_llm_query(self, query: str, api_key: str) -> Dict[str, Any]:
        """AI reasoning and strategic analysis using Google Gemini."""
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            system_context = (
                "You are an expert commercial sales analyst for RiceTec (US commercial rice seed operations).\n"
                "You have access to the RiceTec Sales & Opportunity dataset containing 2,415 records across 2,354 unique customer accounts.\n\n"
                f"Summary KPIs:\n{str(self.de.kpis)}\n\n"
                "Key Columns & Metrics:\n"
                "- 2025 Seed Technologies: 2025 Full Page (678,820 acres, 65.8%), 2025 Non-HT (223,580 acres, 21.7%), 2025 Max-Ace (94,799 acres, 9.2%), 2025 Non-HT MG (34,748 acres, 3.4%).\n"
                "- Historical progression: 2023 Acres (1,396,774), 2024 Acres (1,386,265), 2025 Acres (1,031,946).\n"
                "- Market Opportunity: Total Oppor is 1,775,110 acres.\n"
                "- Geographic footprint: 4 Regions (R01 to R04), 16 Districts (D01 to D16) across 9 States (AR, LA, MO, TX, MS, TN, IL, GA, FL).\n"
                "- Customer Programs: Super Loyalty (507 accounts), Loyalty Rebate (670 accounts), Volume Rebate (490 accounts).\n\n"
                "Guidelines:\n"
                "- Provide clear, professional, executive-ready answers with exact numbers and business insights.\n"
                "- Format using clean Markdown with bolding, lists, and tables when helpful.\n"
            )
            candidate_models = [
                "gemini-3-flash-preview",
                "gemini-3.8-flash",
                "gemini-3.7-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest"
            ]
            response = None
            used_model = "Gemini Flash"
            last_err = None
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=f"{system_context}\n\nUser Question: {query}"
                    )
                    used_model = model_name
                    break
                except Exception as me:
                    last_err = me
                    continue

            if response is None:
                raise last_err or Exception("Failed to query Gemini models.")

            return {
                "text": f"🤖 **Gemini AI Analysis ({used_model})**\n\n{response.text}",
                "table": None,
                "chart": None,
                "metric_cards": [
                    {"label": "Engine", "value": f"Gemini ({used_model})"},
                    {"label": "Mode", "value": "AI Reasoning"}
                ]
            }
        except Exception as e:
            return {
                "text": f"⚠️ Gemini AI call encountered an issue: {str(e)}\n\nFalling back to deterministic engine:\n\n" + self._handle_smart_fallback(query)["text"],
                "table": None,
                "chart": None,
                "metric_cards": None
            }
