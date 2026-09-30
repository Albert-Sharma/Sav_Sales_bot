"""
knowledge_bank.py - 157 Ground-Truth Benchmarks, Dynamic Evaluator Engine,
and Step-by-Step Methodology Resolver for RiceTec US Sales & Opportunity dataset.
"""

import os
import re
import json
import difflib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Load benchmark definitions
BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "benchmark_processed.json")
if not os.path.exists(BENCHMARK_FILE):
    # Fallback to current working directory
    BENCHMARK_FILE = "benchmark_processed.json"

try:
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        BENCHMARK_LIST: List[Dict[str, Any]] = json.load(f)
except Exception:
    BENCHMARK_LIST = []

BENCHMARK_BY_ID: Dict[int, Dict[str, Any]] = {int(b["ID"]): b for b in BENCHMARK_LIST if "ID" in b and str(b["ID"]).isdigit()}


def evaluate_benchmark(bid: int, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Dynamically calculates the answer for a benchmark ID against the live DataFrame.
    Returns calculated answer, step-by-step methodology, reusable dynamic rule, and metrics.
    """
    if bid not in BENCHMARK_BY_ID:
        raise ValueError(f"Unknown benchmark ID: {bid}")

    bench = BENCHMARK_BY_ID[bid]
    q = bench["Question"]
    cat = bench["Category"]
    diff_lvl = bench.get("Difficulty", "Medium")
    exact_ref = bench["Exact Answer"]
    steps = bench["Step-by-Step Calculation"]
    rule = bench["Reusable Dynamic Rule"]

    # Ensure numeric columns are cleaned with 0.0 fill
    num_cols = [
        '2023 Acres', '2024 Acres', '2025 Acres',
        'Max_Acres_Per_Customer', 'Max_Yearly_Acres_Per_Customer',
        'Maxp', 'Max_Acres', 'Oppor',
        '2025 Full Page', '2025 Max-Ace', '2025 Non-HT', '2025 Non-HT MG'
    ]
    work_df = df.copy()
    for col in num_cols:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors='coerce').fillna(0.0)

    # Normalize rebate columns
    for col in ['Loyalty Rebate Status', 'Volume Rebate Status', 'Super Loyalty']:
        if col in work_df.columns:
            work_df[col] = work_df[col].astype(str).str.strip().str.capitalize()
            work_df[col] = work_df[col].replace({'Nan': 'Blank', 'None': 'Blank', '': 'Blank', 'False': 'No', 'True': 'Yes'})

    # State column
    if 'State' not in work_df.columns and 'CountyState' in work_df.columns:
        work_df['State'] = work_df['CountyState'].apply(lambda cs: str(cs).rsplit(',', 1)[-1].strip().upper() if ',' in str(cs) else "")

    computed_answer = ""
    metrics: Dict[str, Any] = {}

    # Category 1: Portfolio (1-20)
    if bid == 1:
        val = len(work_df[work_df['Customer_Group__c'].astype(str).str.strip() != ''])
        computed_answer = f"{val} customers."
        metrics = {"count": val}
    elif bid == 2:
        val = work_df['2023 Acres'].sum()
        computed_answer = f"{val:,.2f} acres."
        metrics = {"acres": val}
    elif bid == 3:
        val = work_df['2024 Acres'].sum()
        computed_answer = f"{val:,.2f} acres."
        metrics = {"acres": val}
    elif bid == 4:
        val = work_df['2025 Acres'].sum()
        computed_answer = f"{val:,.2f} acres."
        metrics = {"acres": val}
    elif bid == 5:
        y23 = work_df['2023 Acres'].sum()
        y24 = work_df['2024 Acres'].sum()
        pct = (y24 / y23 - 1) * 100 if y23 > 0 else 0.0
        diff = y24 - y23
        computed_answer = f"{pct:+.2f}%, a change of {diff:+,.2f} acres."
        metrics = {"growth_pct": pct, "acre_change": diff}
    elif bid == 6:
        y24 = work_df['2024 Acres'].sum()
        y25 = work_df['2025 Acres'].sum()
        pct = (y25 / y24 - 1) * 100 if y24 > 0 else 0.0
        diff = y25 - y24
        computed_answer = f"{pct:+.2f}%, a change of {diff:+,.2f} acres."
        metrics = {"growth_pct": pct, "acre_change": diff}
    elif bid == 7:
        val = work_df['Oppor'].sum()
        computed_answer = f"{val:,.2f} acres."
        metrics = {"opportunity_acres": val}
    elif bid == 8:
        val = work_df['Oppor'].sum() - work_df['2025 Acres'].sum()
        computed_answer = f"{val:,.2f} acres."
        metrics = {"unrealized_acres": val}
    elif bid == 9:
        opp = work_df['Oppor'].sum()
        pct = (work_df['2025 Acres'].sum() / opp * 100) if opp > 0 else 0.0
        computed_answer = f"{pct:.2f}%."
        metrics = {"realized_pct": pct}
    elif bid == 10:
        val = work_df['2025 Acres'].sum() / len(work_df) if len(work_df) > 0 else 0.0
        computed_answer = f"{val:.2f} acres."
        metrics = {"avg_acres": val}
    elif bid == 11:
        val = int((work_df['2025 Acres'] > 0).sum())
        computed_answer = f"{val} customers."
        metrics = {"positive_customers": val}
    elif bid == 12:
        val = int((work_df['2025 Acres'] == 0).sum())
        computed_answer = f"{val} customers."
        metrics = {"zero_customers": val}
    elif bid == 13:
        val = int((work_df['2025 Acres'] > work_df['2024 Acres']).sum())
        computed_answer = f"{val} customers."
        metrics = {"grew_customers": val}
    elif bid == 14:
        val = int((work_df['2025 Acres'] < work_df['2024 Acres']).sum())
        computed_answer = f"{val} customers."
        metrics = {"declined_customers": val}
    elif bid == 15:
        val = int((work_df['2025 Acres'] == work_df['2024 Acres']).sum())
        computed_answer = f"{val} customers."
        metrics = {"unchanged_customers": val}
    elif bid == 16:
        val = int(((work_df['2023 Acres'] > 0) & (work_df['2024 Acres'] > work_df['2023 Acres']) & (work_df['2025 Acres'] > work_df['2024 Acres'])).sum())
        computed_answer = f"{val} customers."
        metrics = {"consecutive_growers": val}
    elif bid == 17:
        val = int((work_df['2025 Acres'] >= 1000).sum())
        computed_answer = f"{val} customers."
        metrics = {"large_growers": val}
    elif bid == 18:
        val = int(((work_df['2025 Acres'] >= 1000) & (work_df['2025 Acres'] < work_df['2024 Acres'])).sum())
        computed_answer = f"{val} customers."
        metrics = {"large_growers_declined": val}
    elif bid == 19:
        val = int(((work_df['2023 Acres'] == 0) & (work_df['2024 Acres'] == 0) & (work_df['2025 Acres'] > 0)).sum())
        computed_answer = f"{val} customers."
        metrics = {"new_growers_2025": val}
    elif bid == 20:
        val = int(((work_df['2024 Acres'] > 0) & (work_df['2025 Acres'] == 0)).sum())
        computed_answer = f"{val} customers."
        metrics = {"fell_to_zero_growers": val}

    # Category 2: Region (21-38)
    elif 21 <= bid <= 36:
        m = re.search(r'R\d{2}', q)
        r_num = m.group(0) if m else "R02"
        m_df = work_df[work_df['Region_Name__c'] == r_num]
        if 'how many customers' in q.lower():
            val = len(m_df)
            computed_answer = f"{val} customers."
            metrics = {"region": r_num, "customers": val}
        elif 'total 2025 acres' in q.lower():
            val = m_df['2025 Acres'].sum()
            computed_answer = f"{val:,.2f} acres."
            metrics = {"region": r_num, "acres_2025": val}
        elif 'weighted 2025 growth' in q.lower():
            y24 = m_df['2024 Acres'].sum()
            y25 = m_df['2025 Acres'].sum()
            if y24 > 0:
                pct = (y25 / y24 - 1) * 100
                computed_answer = f"{pct:.2f}%."
            else:
                computed_answer = "-100.00%."
            metrics = {"region": r_num, "growth_pct": computed_answer}
        elif 'opportunity gap' in q.lower():
            val = m_df['Oppor'].sum() - m_df['2025 Acres'].sum()
            computed_answer = f"{val:,.2f} acres."
            metrics = {"region": r_num, "opp_gap": val}
    elif bid == 37:
        grp = work_df.groupby('Region_Name__c')['2025 Acres'].sum().sort_values(ascending=False)
        top_r = grp.index[0]
        top_v = grp.iloc[0]
        computed_answer = f"{top_r} with {top_v:,.2f} acres."
        metrics = {"top_region": top_r, "acres": top_v}
    elif bid == 38:
        grp = (work_df.groupby('Region_Name__c')['Oppor'].sum() - work_df.groupby('Region_Name__c')['2025 Acres'].sum()).sort_values(ascending=False)
        top_r = grp.index[0]
        top_v = grp.iloc[0]
        computed_answer = f"{top_r} with {top_v:,.2f} acres."
        metrics = {"top_gap_region": top_r, "gap_acres": top_v}

    # Category 3: District (39-86)
    elif 39 <= bid <= 86:
        m = re.search(r'D\d{2}', q)
        dist = m.group(0) if m else "D07"
        m_df = work_df[work_df['District_Name__c'] == dist]
        if 'total 2025 acres' in q.lower():
            val = m_df['2025 Acres'].sum()
            cnt = len(m_df)
            computed_answer = f"{val:,.2f} acres across {cnt} customers."
            metrics = {"district": dist, "acres_2025": val, "customers": cnt}
        elif 'weighted 2025 growth' in q.lower():
            y24 = m_df['2024 Acres'].sum()
            y25 = m_df['2025 Acres'].sum()
            if y24 > 0:
                pct = (y25 / y24 - 1) * 100
                computed_answer = f"{pct:.2f}%."
            else:
                computed_answer = "-100.00%."
            metrics = {"district": dist, "growth_pct": computed_answer}
        elif 'opportunity gap' in q.lower():
            val = m_df['Oppor'].sum() - m_df['2025 Acres'].sum()
            computed_answer = f"{val:,.2f} acres."
            metrics = {"district": dist, "opp_gap": val}

    # Category 4: State (87-113)
    elif 87 <= bid <= 113:
        words = q.replace('?', '').split()
        st = words[-1]
        m_df = work_df[work_df['State'] == st]
        if 'total 2025 acres' in q.lower():
            val = m_df['2025 Acres'].sum()
            cnt = len(m_df)
            computed_answer = f"{val:,.2f} acres across {cnt} customers."
            metrics = {"state": st, "acres_2025": val, "customers": cnt}
        elif 'weighted 2025 growth' in q.lower():
            y24 = m_df['2024 Acres'].sum()
            y25 = m_df['2025 Acres'].sum()
            if y24 > 0:
                pct = (y25 / y24 - 1) * 100
                computed_answer = f"{pct:.2f}%."
            elif y25 == 0 and y24 == 0:
                computed_answer = "Not applicable."
            else:
                computed_answer = "-100.00%."
            metrics = {"state": st, "growth": computed_answer}
        elif 'percentage of portfolio 2025 acres' in q.lower():
            tot = work_df['2025 Acres'].sum()
            st_tot = m_df['2025 Acres'].sum()
            pct = (st_tot / tot * 100) if tot > 0 else 0.0
            computed_answer = f"{pct:.2f}%."
            metrics = {"state": st, "portfolio_share_pct": pct}

    # Category 5: Customer Rankings (114-133)
    elif 114 <= bid <= 123:  # Rank #1 to #10 by 2025 acres
        rank = bid - 113
        sorted_df = work_df.sort_values(by='2025 Acres', ascending=False).reset_index(drop=True)
        row = sorted_df.iloc[rank - 1]
        cust = row['Customer_Group__c']
        ac = row['2025 Acres']
        cs = row['CountyState']
        dist = row['District_Name__c']
        reg = row['Region_Name__c']
        computed_answer = f"{cust} with {ac:,.2f} acres ({cs}, {dist}, {reg})."
        metrics = {"rank": rank, "customer": cust, "acres": ac, "county_state": cs, "district": dist, "region": reg}
    elif 124 <= bid <= 128:  # Largest acreage increase #1 to #5
        rank = bid - 123
        work_df['change_24_25'] = work_df['2025 Acres'] - work_df['2024 Acres']
        sorted_df = work_df.sort_values(by='change_24_25', ascending=False).reset_index(drop=True)
        row = sorted_df.iloc[rank - 1]
        cust = row['Customer_Group__c']
        chg = row['change_24_25']
        y24 = row['2024 Acres']
        y25 = row['2025 Acres']
        computed_answer = f"{cust}: +{chg:,.2f} acres, from {y24:,.2f} to {y25:,.2f}."
        metrics = {"rank": rank, "customer": cust, "change": chg, "y2024": y24, "y2025": y25}
    elif 129 <= bid <= 133:  # Largest acreage decline #1 to #5
        rank = bid - 128
        work_df['change_24_25'] = work_df['2025 Acres'] - work_df['2024 Acres']
        sorted_df = work_df.sort_values(by='change_24_25', ascending=True).reset_index(drop=True)
        row = sorted_df.iloc[rank - 1]
        cust = row['Customer_Group__c']
        chg = row['change_24_25']
        y24 = row['2024 Acres']
        y25 = row['2025 Acres']
        computed_answer = f"{cust}: {chg:,.2f} acres, from {y24:,.2f} to {y25:,.2f}."
        metrics = {"rank": rank, "customer": cust, "change": chg, "y2024": y24, "y2025": y25}

    # Category 6: Opportunity (134-138)
    elif 134 <= bid <= 138:  # Largest opportunity gap #1 to #5
        rank = bid - 133
        work_df['gap'] = work_df['Oppor'] - work_df['2025 Acres']
        sorted_df = work_df.sort_values(by='gap', ascending=False).reset_index(drop=True)
        row = sorted_df.iloc[rank - 1]
        cust = row['Customer_Group__c']
        gap = row['gap']
        computed_answer = f"{cust} with a {gap:,.2f} acre gap."
        metrics = {"rank": rank, "customer": cust, "gap": gap}

    # Category 7: Product Mix (139-142)
    elif bid == 139:
        val = work_df['2025 Full Page'].sum()
        tot = work_df['2025 Acres'].sum()
        pct = (val / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{val:,.2f} acres, representing {pct:.2f}% of 2025 acres."
        metrics = {"acres": val, "share_pct": pct}
    elif bid == 140:
        val = work_df['2025 Max-Ace'].sum()
        tot = work_df['2025 Acres'].sum()
        pct = (val / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{val:,.2f} acres, representing {pct:.2f}% of 2025 acres."
        metrics = {"acres": val, "share_pct": pct}
    elif bid == 141:
        val = work_df['2025 Non-HT'].sum()
        tot = work_df['2025 Acres'].sum()
        pct = (val / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{val:,.2f} acres, representing {pct:.2f}% of 2025 acres."
        metrics = {"acres": val, "share_pct": pct}
    elif bid == 142:
        val = work_df['2025 Non-HT MG'].sum()
        tot = work_df['2025 Acres'].sum()
        pct = (val / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{val:,.2f} acres, representing {pct:.2f}% of 2025 acres."
        metrics = {"acres": val, "share_pct": pct}

    # Category 8: Data Quality (143)
    elif bid == 143:
        diffs = (work_df['2025 Full Page'] + work_df['2025 Max-Ace'] + work_df['2025 Non-HT'] + work_df['2025 Non-HT MG'] - work_df['2025 Acres']).abs()
        mismatches = int((diffs > 0.01).sum())
        if mismatches == 0:
            computed_answer = f"Yes. All {len(work_df)} rows reconcile within 0.01 acre."
        else:
            computed_answer = f"No. {mismatches} rows differ by more than 0.01 acre."
        metrics = {"mismatches": mismatches, "total_rows": len(work_df)}

    # Category 9: Loyalty & Rebate Status (144-155)
    elif bid == 144:
        sub = work_df[work_df['Super Loyalty'] == 'Yes']
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 145:
        sub = work_df[work_df['Super Loyalty'] == 'No']
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 146:
        sub = work_df[work_df['Super Loyalty'] == 'Blank']
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 147:
        sub = work_df[work_df['Super Loyalty'] == 'Yes']
        tot = work_df['2025 Acres'].sum()
        ac = sub['2025 Acres'].sum()
        pct = (ac / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{pct:.2f}%, or {ac:,.2f} acres."
        metrics = {"acres": ac, "share_pct": pct}
    elif bid == 148:
        sub_y = work_df[work_df['Super Loyalty'] == 'Yes']
        sub_n = work_df[work_df['Super Loyalty'] == 'No']
        avg_y = sub_y['2025 Acres'].sum() / len(sub_y) if len(sub_y) > 0 else 0.0
        avg_n = sub_n['2025 Acres'].sum() / len(sub_n) if len(sub_n) > 0 else 0.0
        mult = (avg_y / avg_n) if avg_n > 0 else 0.0
        computed_answer = f"Yes: {avg_y:,.2f} acres; No: {avg_n:,.2f} acres; Yes is {mult:.2f}x higher."
        metrics = {"avg_yes": avg_y, "avg_no": avg_n, "multiplier": mult}
    elif bid == 149:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Yes') & (work_df['Volume Rebate Status'] == 'Yes') & (work_df['Super Loyalty'] == 'Yes')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 150:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Yes') & (work_df['Volume Rebate Status'] == 'Yes') & (work_df['Super Loyalty'] == 'No')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 151:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Yes') & (work_df['Volume Rebate Status'] == 'No') & (work_df['Super Loyalty'] == 'Yes')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 152:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Yes') & (work_df['Volume Rebate Status'] == 'No') & (work_df['Super Loyalty'] == 'No')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 153:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'No') & (work_df['Volume Rebate Status'] == 'No') & (work_df['Super Loyalty'] == 'No')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 154:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Blank') & (work_df['Volume Rebate Status'] == 'Blank') & (work_df['Super Loyalty'] == 'No')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}
    elif bid == 155:
        sub = work_df[(work_df['Loyalty Rebate Status'] == 'Blank') & (work_df['Volume Rebate Status'] == 'Blank') & (work_df['Super Loyalty'] == 'Blank')]
        cnt = len(sub)
        ac = sub['2025 Acres'].sum()
        computed_answer = f"{cnt} customers totaling {ac:,.2f} 2025 acres."
        metrics = {"count": cnt, "acres": ac}

    # Category 10: Business Insight (156-157)
    elif bid == 156:
        tot = work_df['2025 Acres'].sum()
        top10 = work_df.sort_values(by='2025 Acres', ascending=False).head(10)['2025 Acres'].sum()
        pct = (top10 / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{pct:.2f}%, or {top10:,.2f} acres."
        metrics = {"acres": top10, "share_pct": pct}
    elif bid == 157:
        tot = work_df['2025 Acres'].sum()
        top50 = work_df.sort_values(by='2025 Acres', ascending=False).head(50)['2025 Acres'].sum()
        pct = (top50 / tot * 100) if tot > 0 else 0.0
        computed_answer = f"{pct:.2f}%, or {top50:,.2f} acres."
        metrics = {"acres": top50, "share_pct": pct}
    else:
        computed_answer = exact_ref

    reconciled = (computed_answer.strip() == exact_ref.strip())

    return {
        "id": bid,
        "category": cat,
        "difficulty": diff_lvl,
        "question": q,
        "computed_answer": computed_answer,
        "step_by_step_calculation": steps,
        "reusable_dynamic_rule": rule,
        "reference_exact_answer": exact_ref,
        "reconciled": reconciled,
        "metrics": metrics,
        "total_records_evaluated": len(work_df)
    }


def format_benchmark_response(eval_res: Dict[str, Any]) -> str:
    """
    Formats the evaluation result into user-facing markdown explaining:
    1. Dynamic Result
    2. Step-by-Step Calculation Methodology
    3. Reusable Dynamic Rule / Formula
    """
    bid = eval_res["id"]
    ans = eval_res["computed_answer"]
    steps = eval_res["step_by_step_calculation"]
    rule = eval_res["reusable_dynamic_rule"]
    cat = eval_res["category"]
    rec_count = eval_res.get("total_records_evaluated", 2415)

    # Format step by step into bullet points if not already numbered cleanly
    formatted_steps = steps
    # If steps contain '1) ... 2) ...' break them onto separate lines for readability
    if "1)" in steps and "2)" in steps:
        formatted_steps = re.sub(r'(\d+\))', r'\n\1', steps).strip()

    md = (
        f"### 📊 Dynamic Calculation Result\n"
        f"**{ans}**\n\n"
        f"### 📋 Step-by-Step Calculation Methodology\n"
        f"{formatted_steps}\n\n"
        f"### 📐 Reusable Dynamic Rule / Formula\n"
        f"```excel\n{rule}\n```\n\n"
        f"---\n"
        f"*Category: {cat} | Benchmark Rule ID #{bid} | Evaluated dynamically on live dataset ({rec_count:,} records)*"
    )
    return md


def resolve_benchmark_query(query: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Intelligently resolves a natural language query to one of the 157 benchmark rules,
    dynamically evaluates it against the live DataFrame, and returns the result dictionary.
    """
    q_clean = query.strip().lower()
    q_norm = re.sub(r'[^a-z0-9\s=]', ' ', q_clean)
    q_tokens = set(q_norm.split())

    # --- Step A: Check for Tri-Rebate and Loyalty Combinations first (IDs 144-155) ---
    has_super = 'super' in q_clean or 'super loyalty' in q_clean
    has_volume = 'volume' in q_clean or 'volume rebate' in q_clean
    has_loyalty = 'loyalty' in q_clean or 'loyalty rebate' in q_clean

    # Super Loyalty alone
    if has_super and not has_volume:
        if 'blank' in q_clean or 'null' in q_clean or 'missing' in q_clean or 'nan' in q_clean:
            return evaluate_benchmark(146, df)
        if 'yes' in q_clean and ('share' in q_clean or 'percentage' in q_clean or 'portfolio' in q_clean):
            return evaluate_benchmark(147, df)
        if 'average' in q_clean or 'compare' in q_clean or 'higher' in q_clean or 'versus' in q_clean or 'vs' in q_clean:
            return evaluate_benchmark(148, df)
        if 'yes' in q_clean:
            return evaluate_benchmark(144, df)
        if 'no' in q_clean:
            return evaluate_benchmark(145, df)

    # Tri-Rebate combinations (IDs 149-155)
    if (has_loyalty or has_volume or has_super) and ('=' in q_clean or 'and' in q_clean or 'but' in q_clean):
        # Extract explicit status for loyalty, volume, super
        # Check loyalty
        m_l = re.search(r'loyalty\s*(?:rebate)?\s*(?:status)?\s*(?:=|\s+is\s+)?\s*(yes|no|blank)', q_clean)
        # Check volume
        m_v = re.search(r'volume\s*(?:rebate)?\s*(?:status)?\s*(?:=|\s+is\s+)?\s*(yes|no|blank)', q_clean)
        # Check super
        m_s = re.search(r'super\s*(?:loyalty)?\s*(?:status)?\s*(?:=|\s+is\s+)?\s*(yes|no|blank)', q_clean)

        l_val = m_l.group(1).capitalize() if m_l else None
        v_val = m_v.group(1).capitalize() if m_v else None
        s_val = m_s.group(1).capitalize() if m_s else None

        if l_val == 'Yes' and v_val == 'Yes' and s_val == 'Yes':
            return evaluate_benchmark(149, df)
        if l_val == 'Yes' and v_val == 'Yes' and s_val == 'No':
            return evaluate_benchmark(150, df)
        if l_val == 'Yes' and v_val == 'No' and s_val == 'Yes':
            return evaluate_benchmark(151, df)
        if l_val == 'Yes' and v_val == 'No' and s_val == 'No':
            return evaluate_benchmark(152, df)
        if l_val == 'No' and v_val == 'No' and s_val == 'No':
            return evaluate_benchmark(153, df)
        if l_val == 'Blank' and v_val == 'Blank' and s_val == 'No':
            return evaluate_benchmark(154, df)
        if l_val == 'Blank' and v_val == 'Blank' and s_val == 'Blank':
            return evaluate_benchmark(155, df)

        # Handle queries like "How many have Super Loyalty=Yes but Volume Rebate=No?"
        if s_val == 'Yes' and v_val == 'No':
            # This directly corresponds to ID 151 (Loyalty=Yes, Volume=No, Super=Yes)
            return evaluate_benchmark(151, df)

    # --- Step B: Concentration & Business Insights (IDs 156-157) ---
    if 'top 10' in q_clean and ('share' in q_clean or 'held' in q_clean or 'concentration' in q_clean or 'percentage' in q_clean or '%' in q_clean or 'acres' in q_clean):
        return evaluate_benchmark(156, df)
    if 'top 50' in q_clean and ('share' in q_clean or 'held' in q_clean or 'concentration' in q_clean or 'percentage' in q_clean or '%' in q_clean or 'acres' in q_clean):
        return evaluate_benchmark(157, df)

    # --- Step C: Data Quality & Reconciliation (ID 143) ---
    if 'reconcile' in q_clean or 'categories reconcile' in q_clean or 'product reconciliation' in q_clean:
        return evaluate_benchmark(143, df)

    # --- Step D: Product Mix (IDs 139-142) ---
    if 'full page' in q_clean:
        return evaluate_benchmark(139, df)
    if 'max-ace' in q_clean or 'max ace' in q_clean:
        return evaluate_benchmark(140, df)
    if 'non-ht mg' in q_clean or 'non ht mg' in q_clean:
        return evaluate_benchmark(142, df)
    if 'non-ht' in q_clean or 'non ht' in q_clean:
        return evaluate_benchmark(141, df)

    # --- Step E: Customer Rankings & Transitions (IDs 114-138) ---
    # Top N by 2025 acres (IDs 114-123)
    m_rank = re.search(r'ranks?\s*#?(\d+)\s*(?:by\s*2025\s*acres|in\s*2025)?', q_clean)
    if m_rank and not ('increase' in q_clean or 'decline' in q_clean or 'opportunity' in q_clean or 'gap' in q_clean):
        r_num = int(m_rank.group(1))
        if 1 <= r_num <= 10:
            return evaluate_benchmark(113 + r_num, df)

    # Acreage Increase (IDs 124-128)
    if 'increase' in q_clean or 'growth' in q_clean or 'gainer' in q_clean or 'largest acreage increase' in q_clean:
        m_inc = re.search(r'#?(\d+)', q_clean)
        rank = int(m_inc.group(1)) if m_inc else 1
        if 1 <= rank <= 5:
            return evaluate_benchmark(123 + rank, df)

    # Acreage Decline (IDs 129-133)
    if 'decline' in q_clean or 'decrease' in q_clean or 'lost' in q_clean or 'largest acreage decline' in q_clean:
        m_dec = re.search(r'#?(\d+)', q_clean)
        rank = int(m_dec.group(1)) if m_dec else 1
        if 1 <= rank <= 5:
            return evaluate_benchmark(128 + rank, df)

    # Largest Opportunity Gap Customer (IDs 134-138)
    if ('opportunity gap' in q_clean or 'largest opportunity' in q_clean or 'largest gap' in q_clean) and ('who' in q_clean or 'customer' in q_clean or 'ranks' in q_clean):
        m_gap = re.search(r'#?(\d+)', q_clean)
        rank = int(m_gap.group(1)) if m_gap else 1
        if 1 <= rank <= 5:
            return evaluate_benchmark(133 + rank, df)

    # --- Step F: District Metrics (IDs 39-86) ---
    m_dist = re.search(r'\b(d0[1-9]|d1[0-6])\b', q_clean)
    if m_dist:
        dist = m_dist.group(1).upper()
        # Find district benchmarks
        dist_benchmarks = [b for b in BENCHMARK_LIST if 39 <= int(b["ID"]) <= 86 and dist in b["Question"]]
        if 'total' in q_clean and 'acres' in q_clean or 'how many acres' in q_clean or 'customer' in q_clean and 'total' in q_clean:
            for b in dist_benchmarks:
                if 'total 2025 acres' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)
        elif 'growth' in q_clean or 'weighted' in q_clean:
            for b in dist_benchmarks:
                if 'weighted 2025 growth' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)
        elif 'gap' in q_clean or 'opportunity' in q_clean:
            for b in dist_benchmarks:
                if 'opportunity gap' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)

    # --- Step G: Region Metrics (IDs 21-38) ---
    if 'region leads in 2025 acres' in q_clean or 'which region leads' in q_clean:
        return evaluate_benchmark(37, df)
    if 'region has the largest opportunity gap' in q_clean or 'largest opportunity gap' in q_clean and 'region' in q_clean:
        return evaluate_benchmark(38, df)

    m_reg = re.search(r'\b(r0[1-4])\b', q_clean)
    if m_reg:
        reg = m_reg.group(1).upper()
        reg_benchmarks = [b for b in BENCHMARK_LIST if 21 <= int(b["ID"]) <= 36 and reg in b["Question"]]
        if 'how many customers' in q_clean or 'customer count' in q_clean or 'customers are in' in q_clean:
            for b in reg_benchmarks:
                if 'how many customers' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)
        elif 'total 2025 acres' in q_clean or 'total acres' in q_clean:
            for b in reg_benchmarks:
                if 'total 2025 acres' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)
        elif 'growth' in q_clean or 'weighted' in q_clean:
            for b in reg_benchmarks:
                if 'weighted 2025 growth' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)
        elif 'gap' in q_clean or 'opportunity' in q_clean:
            for b in reg_benchmarks:
                if 'opportunity gap' in b["Question"].lower():
                    return evaluate_benchmark(int(b["ID"]), df)

    # --- Step H: State Metrics (IDs 87-113) ---
    states = ['AR', 'LA', 'MO', 'MS', 'TX', 'IL', 'TN', 'GA', 'FL']
    for st in states:
        # Match state as word boundary
        if re.search(rf'\b{st.lower()}\b', q_clean):
            st_benchmarks = [b for b in BENCHMARK_LIST if 87 <= int(b["ID"]) <= 113 and b["Question"].endswith(f"{st}?") or f"in {st}?" in b["Question"]]
            if 'percentage' in q_clean or 'share' in q_clean or 'portfolio' in q_clean:
                for b in st_benchmarks:
                    if 'percentage of portfolio' in b["Question"].lower():
                        return evaluate_benchmark(int(b["ID"]), df)
            elif 'growth' in q_clean or 'weighted' in q_clean:
                for b in st_benchmarks:
                    if 'weighted 2025 growth' in b["Question"].lower():
                        return evaluate_benchmark(int(b["ID"]), df)
            elif 'total' in q_clean or 'acres' in q_clean or 'how many' in q_clean:
                for b in st_benchmarks:
                    if 'total 2025 acres' in b["Question"].lower():
                        return evaluate_benchmark(int(b["ID"]), df)

    # --- Step I: Portfolio Level Metrics (IDs 1-20) ---
    if 'consecutive' in q_clean or 'both 2024 and 2025' in q_clean or 'both consecutive years' in q_clean:
        return evaluate_benchmark(16, df)
    if 'new in 2025' in q_clean or 'new customers' in q_clean or 'zero acres in 2023 and 2024' in q_clean:
        return evaluate_benchmark(19, df)
    if 'fell from positive 2024' in q_clean or 'dropped to zero' in q_clean or 'fell to zero' in q_clean:
        return evaluate_benchmark(20, df)
    if '1 000 acres declined' in q_norm or '1000 acres declined' in q_clean or 'at least 1,000 acres declined' in q_clean:
        return evaluate_benchmark(18, df)
    if 'at least 1 000 acres' in q_norm or 'at least 1,000 acres' in q_clean or '>= 1000 acres' in q_clean or '1000 or more acres' in q_clean:
        return evaluate_benchmark(17, df)
    if 'positive 2025 acres' in q_clean or 'positive acres' in q_clean:
        return evaluate_benchmark(11, df)
    if 'zero 2025 acres' in q_clean or 'zero acres in 2025' in q_clean:
        return evaluate_benchmark(12, df)
    if 'unchanged from 2024 to 2025' in q_clean or 'customers were unchanged' in q_clean:
        return evaluate_benchmark(15, df)
    if 'declined from 2024 to 2025' in q_clean or 'customers declined' in q_clean:
        return evaluate_benchmark(14, df)
    if 'grew from 2024 to 2025' in q_clean or 'customers grew' in q_clean:
        return evaluate_benchmark(13, df)
    if 'average 2025 acreage' in q_clean or 'average acreage per customer' in q_clean:
        return evaluate_benchmark(10, df)
    if 'percentage of opportunity was realized' in q_clean or 'opportunity was realized' in q_clean or 'realized in 2025' in q_clean:
        return evaluate_benchmark(9, df)
    if 'unrealized opportunity' in q_clean:
        return evaluate_benchmark(8, df)
    if 'total sales opportunity' in q_clean or 'total opportunity' in q_clean:
        return evaluate_benchmark(7, df)
    if 'weighted growth from 2024 to 2025' in q_clean or 'growth from 2024 to 2025' in q_clean:
        return evaluate_benchmark(6, df)
    if 'weighted growth from 2023 to 2024' in q_clean or 'growth from 2023 to 2024' in q_clean:
        return evaluate_benchmark(5, df)
    if 'total acres in 2025' in q_clean or '2025 total acres' in q_clean:
        return evaluate_benchmark(4, df)
    if 'total acres in 2024' in q_clean or '2024 total acres' in q_clean:
        return evaluate_benchmark(3, df)
    if 'total acres in 2023' in q_clean or '2023 total acres' in q_clean:
        return evaluate_benchmark(2, df)
    if 'customer records' in q_clean or 'how many customer records' in q_clean or 'number of customer records' in q_clean:
        return evaluate_benchmark(1, df)

    # --- Step J: Fuzzy match against all 157 benchmark questions ---
    questions = [b["Question"] for b in BENCHMARK_LIST]
    matches = difflib.get_close_matches(query, questions, n=1, cutoff=0.75)
    if matches:
        best_q = matches[0]
        for b in BENCHMARK_LIST:
            if b["Question"] == best_q:
                return evaluate_benchmark(int(b["ID"]), df)

    return None
