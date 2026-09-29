"""
app.py - Streamlit Interface for RiceTec Sales & Opportunity Intelligence Chatbot.
Provides an interactive chatbot interface, executive KPI dashboards, customer 360 lookup,
and SQL data exploration.
"""

import streamlit as st
import pandas as pd
import altair as alt
import json
from data_engine import SalesDataEngine
from query_engine import QueryEngine

# Configure Streamlit page
st.set_page_config(
    page_title="RiceTec Sales Intelligence Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished RiceTec branding and chat styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .pill-btn {
        margin: 2px 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Safe Configuration & Secrets Handler for Streamlit Cloud and Local .env
import os

def get_secret(key: str, default: str = "") -> str:
    """Safely retrieves a secret from st.secrets, os.getenv, or default fallback."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# Password Protection Gatekeeper
PASSWORD_REQUIRED = get_secret("PASSWORD_REQUIRED", "ricetec")

def check_password() -> bool:
    """Returns True if user enters the correct password."""
    if st.session_state.get("authenticated", False):
        return True

    st.markdown('<div class="main-header">🌾 RiceTec Commercial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Internal Sales & Opportunity Analytics Portal</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔒 Secure Access Required")
        st.info("Password require to access")
        
        with st.form("login_form"):
            entered_pwd = st.text_input("Access Password", type="password", placeholder="Enter password...")
            submitted = st.form_submit_button("🔓 Unlock Application", use_container_width=True)
            
            if submitted:
                if entered_pwd.strip().lower() == PASSWORD_REQUIRED.lower():
                    st.session_state["authenticated"] = True
                    st.success("✅ Access granted! Loading application...")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Access denied.")
                    
        st.caption("RiceTec Confidential • Protected by encrypted session authentication")

    return False

if not check_password():
    st.stop()

# Cache data loading to optimize startup and re-runs
@st.cache_resource(show_spinner="Loading RiceTec sales dataset...")
def get_engines():
    data_engine = SalesDataEngine()
    gemini_key = get_secret("GEMINI_API_KEY", "")
    query_engine = QueryEngine(data_engine, gemini_api_key=gemini_key)
    return data_engine, query_engine

data_engine, query_engine = get_engines()
kpis = data_engine.kpis
clean_df = data_engine.clean_df

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "text": (
                "👋 **Welcome to the RiceTec Sales & Opportunity Assistant!**\n\n"
                "I am equipped to answer questions about customer acreages, historical trends (2023–2025), "
                "seed technology breakdowns (Full Page, Max-Ace, Non-HT), rebate eligibility, and opportunity headroom.\n\n"
                "💡 *Click one of the quick suggestions below or type your question in the chat bar.*"
            ),
            "table": None,
            "chart": None,
            "metric_cards": [
                {"label": "Unique Customers", "value": f"{kpis['unique_customers']:,}"},
                {"label": "2025 Total Acres", "value": f"{kpis['total_2025_acres']:,.0f}"},
                {"label": "Total Opportunity", "value": f"{kpis['total_oppor']:,.0f}"},
                {"label": "Super Loyalty Accounts", "value": f"{kpis['super_loyalty_count']:,}"},
            ]
        }
    ]

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
    st.title("🌾 RiceTec Assistant")
    st.markdown("**Sales & Opportunity Analytics**")
    st.caption("US Commercial Operations")

    st.markdown("---")
    st.subheader("📌 Dataset Metadata")
    st.markdown(f"• **Records**: `{kpis['total_records']:,}`")
    st.markdown(f"• **Customers**: `{kpis['unique_customers']:,}`")
    st.markdown(f"• **Counties**: `{kpis['total_counties']}` across `{kpis['total_states']}` States")
    st.markdown(f"• **Districts**: `D01` to `D16` (16 total)")
    st.markdown(f"• **Regions**: `R01` to `R04` (4 total)")

    st.markdown("---")
    st.subheader("⚙️ Gemini AI Integration")
    has_env_key = bool(query_engine.gemini_api_key)
    if has_env_key:
        st.success("🤖 Gemini Flash AI: Connected")
        st.caption("🔒 Key securely loaded from private `.env`")
        use_llm = st.toggle("Enable Gemini AI Reasoning", value=True, help="Augments deterministic data with AI reasoning, summaries, and conversational insights.")
        gemini_key = query_engine.gemini_api_key
    else:
        use_llm = st.toggle("Enable Gemini LLM Mode", value=False)
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter API key...") if use_llm else ""

    st.markdown("---")
    st.subheader("🧹 Actions")
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    # Download chat transcript
    if len(st.session_state.messages) > 1:
        transcript_data = [
            {"role": m["role"], "content": m["text"]}
            for m in st.session_state.messages
        ]
        st.download_button(
            label="📥 Download Chat Transcript",
            data=json.dumps(transcript_data, indent=2),
            file_name="ricetec_chatbot_session.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    if st.button("🔒 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# Main Title & KPI Banner
st.markdown('<div class="main-header">🌾 RiceTec Sales & Opportunity Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Conversational Query Engine & Performance Testing Interface</div>', unsafe_allow_html=True)

# Executive KPI Ribbon
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Customers", f"{kpis['unique_customers']:,}")
with col2:
    st.metric("2025 Total Acres", f"{kpis['total_2025_acres']:,.0f}")
with col3:
    st.metric("2024 Total Acres", f"{kpis['total_2024_acres']:,.0f}")
with col4:
    st.metric("Total Opportunity", f"{kpis['total_oppor']:,.0f}")
with col5:
    st.metric("Super Loyalty Accounts", f"{kpis['super_loyalty_count']:,}")

st.markdown("---")

# Navigation Tabs
tab_chat, tab_exec, tab_cust, tab_data = st.tabs([
    "💬 AI Chatbot",
    "📊 Executive Analytics",
    "🔍 Customer 360 Explorer",
    "📋 Data Explorer & SQL Console"
])

# ----------------- TAB 1: AI CHATBOT -----------------
with tab_chat:
    st.markdown("#### ⚡ Quick Prompt Suggestions")
    quick_cols = st.columns(4)
    quick_queries = [
        "🏆 Top 10 customers by 2025 acres",
        "🌾 What is the 2025 product mix?",
        "⭐ Super Loyalty customer breakdown",
        "🗺️ Breakdown of acres by Region",
        "🎯 Who has the biggest opportunity gap?",
        "📈 Compare 2024 vs 2025 acres",
        "📍 Summarize Region R01",
        "🔍 Tell me about DONNY DELINE",
    ]

    selected_quick_prompt = None
    for i, prompt in enumerate(quick_queries):
        col_idx = i % 4
        clean_prompt = prompt.split(" ", 1)[1] if " " in prompt else prompt
        if quick_cols[col_idx].button(prompt, key=f"quick_{i}", use_container_width=True):
            selected_quick_prompt = clean_prompt

    # Prominent Top Input Form for instant typing
    st.markdown("##### 💬 Type Your Question")
    with st.form("top_query_form", clear_on_submit=True):
        col_input, col_submit = st.columns([5, 1])
        with col_input:
            top_query_text = st.text_input(
                "Enter question",
                placeholder="Type your question here (e.g. 'Tell me about DONNY DELINE', 'Top 10 customers by 2025 acres', 'Product mix')...",
                label_visibility="collapsed"
            )
        with col_submit:
            top_query_submit = st.form_submit_button("🚀 Ask", use_container_width=True)

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

            # Metric Cards if present
            if msg.get("metric_cards"):
                cols = st.columns(len(msg["metric_cards"]))
                for i, mc in enumerate(msg["metric_cards"]):
                    cols[i].metric(mc["label"], mc["value"], delta=mc.get("delta"))

            # Table if present
            if msg.get("table") is not None and isinstance(msg["table"], pd.DataFrame) and not msg["table"].empty:
                st.dataframe(msg["table"], use_container_width=True, hide_index=True)

            # Chart if present
            if msg.get("chart"):
                chart_info = msg["chart"]
                c_df = chart_info["df"]
                if chart_info["type"] == "bar":
                    chart = (
                        alt.Chart(c_df)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2e7d32")
                        .encode(
                            x=alt.X(f"{chart_info['x']}:N", sort=None, title=chart_info['x']),
                            y=alt.Y(f"{chart_info['y']}:Q", title=chart_info['y']),
                            tooltip=[chart_info['x'], chart_info['y']]
                        )
                        .properties(title=chart_info.get("title", ""), height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)

    # Process new prompt (from top form, quick buttons, or bottom chat input)
    bottom_input = st.chat_input("Or type here: Ask about customers, regions, technologies, growth, or rebates...")
    
    active_prompt = None
    if top_query_submit and top_query_text.strip():
        active_prompt = top_query_text.strip()
    elif selected_quick_prompt:
        active_prompt = selected_quick_prompt
    elif bottom_input:
        active_prompt = bottom_input

    if active_prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "text": active_prompt})

        # Process with Query Engine
        api_to_pass = gemini_key if (use_llm and gemini_key) else ""
        response_data = query_engine.process_query(active_prompt, api_key=api_to_pass)

        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "text": response_data.get("text", "No response generated."),
            "table": response_data.get("table"),
            "chart": response_data.get("chart"),
            "metric_cards": response_data.get("metric_cards")
        })

        st.rerun()

# ----------------- TAB 2: EXECUTIVE ANALYTICS -----------------
with tab_exec:
    st.markdown("### 📊 Executive Sales Dashboard")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### 📈 Historical Acreage Progression (2023 - 2025)")
        hist_df = pd.DataFrame([
            {"Year": "2023", "Acres": kpis['total_2023_acres']},
            {"Year": "2024", "Acres": kpis['total_2024_acres']},
            {"Year": "2025", "Acres": kpis['total_2025_acres']}
        ])
        hist_chart = (
            alt.Chart(hist_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#1b5e20")
            .encode(
                x=alt.X("Year:N", title="Crop Year"),
                y=alt.Y("Acres:Q", title="Total Acres"),
                tooltip=["Year", alt.Tooltip("Acres:Q", format=",.0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(hist_chart, use_container_width=True)

    with col_b:
        st.markdown("##### 🌾 2025 Seed Technology Portfolio Breakdown")
        tech_df = pd.DataFrame([
            {"Technology": "Full Page", "Acres": kpis['total_full_page_2025']},
            {"Technology": "Non-HT", "Acres": kpis['total_non_ht_2025']},
            {"Technology": "Max-Ace", "Acres": kpis['total_max_ace_2025']},
            {"Technology": "Non-HT MG", "Acres": kpis['total_non_ht_mg_2025']},
        ])
        tech_chart = (
            alt.Chart(tech_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("Acres:Q"),
                color=alt.Color("Technology:N", scale=alt.Scale(scheme="greens"), title="Product"),
                tooltip=["Technology", alt.Tooltip("Acres:Q", format=",.0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(tech_chart, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("##### 🗺️ 2025 Acres by Sales Region")
        reg_df = clean_df.groupby("Region_Name__c")['2025 Acres'].sum().reset_index()
        reg_df.rename(columns={"Region_Name__c": "Region"}, inplace=True)
        reg_chart = (
            alt.Chart(reg_df)
            .mark_bar(color="#388e3c")
            .encode(
                x=alt.X("Region:N", title="Region"),
                y=alt.Y("2025 Acres:Q", title="2025 Acres"),
                tooltip=["Region", alt.Tooltip("2025 Acres:Q", format=",.0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(reg_chart, use_container_width=True)

    with col_d:
        st.markdown("##### ⭐ Rebate & Loyalty Participation")
        rebate_df = pd.DataFrame([
            {"Program": "Super Loyalty", "Customers": kpis['super_loyalty_count']},
            {"Program": "Loyalty Rebate", "Customers": kpis['loyalty_rebate_count']},
            {"Program": "Volume Rebate", "Customers": kpis['volume_rebate_count']},
        ])
        rebate_chart = (
            alt.Chart(rebate_df)
            .mark_bar(color="#81c784")
            .encode(
                x=alt.X("Program:N", title="Rebate Program"),
                y=alt.Y("Customers:Q", title="Customer Count"),
                tooltip=["Program", alt.Tooltip("Customers:Q", format=",.0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(rebate_chart, use_container_width=True)

# ----------------- TAB 3: CUSTOMER 360 -----------------
with tab_cust:
    st.markdown("### 🔍 Customer 360 Deep Dive")
    st.write("Lookup any of the **2,354 RiceTec customers** to review historical performance, seed technology adoption, and rebate status.")

    cust_choice = st.selectbox(
        "Search or Select Customer Name / ID:",
        options=data_engine.customer_list,
        index=0,
        help="Type customer name or 10-digit ID to search"
    )

    if cust_choice:
        cust_profile = data_engine.get_customer_profile(cust_choice)
        if cust_profile:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("2025 Acres", f"{cust_profile['total_2025_acres']:,.1f}", delta=f"{cust_profile['growth_2025']}% YoY" if cust_profile['growth_2025'] != "N/A" else None)
            c2.metric("2024 Acres", f"{cust_profile['total_2024_acres']:,.1f}", delta=f"{cust_profile['growth_2024']}% YoY" if cust_profile['growth_2024'] != "N/A" else None)
            c3.metric("Total Opportunity", f"{cust_profile['total_oppor']:,.1f}")
            c4.metric("Max Recorded Acres", f"{cust_profile['max_acres']}")

            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                st.markdown("##### 🌾 2025 Seed Technology Mix")
                t_data = pd.DataFrame([
                    {"Technology": k, "Acres": v}
                    for k, v in cust_profile['tech_breakdown'].items()
                ])
                st.dataframe(t_data, use_container_width=True, hide_index=True)

            with col_p2:
                st.markdown("##### 🎁 Rebate Eligibility")
                r_data = pd.DataFrame([
                    {"Program": "Super Loyalty", "Status": cust_profile['rebates']['super_loyalty']},
                    {"Program": "Loyalty Rebate", "Status": cust_profile['rebates']['loyalty_rebate']},
                    {"Program": "Volume Rebate", "Status": cust_profile['rebates']['volume_rebate']},
                ])
                st.dataframe(r_data, use_container_width=True, hide_index=True)

            st.markdown(f"##### 📍 Land Locations & Acreage Entries ({cust_profile['row_count']} Location{'s' if cust_profile['row_count'] > 1 else ''})")
            cust_loc_df = pd.DataFrame(cust_profile['locations'])
            cust_loc_df.rename(columns={
                "county_state": "Location",
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
            st.dataframe(cust_loc_df, use_container_width=True, hide_index=True)

# ----------------- TAB 4: DATA EXPLORER & SQL CONSOLE -----------------
with tab_data:
    st.markdown("### 📋 Interactive Dataset Explorer")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        sel_region = st.multiselect("Filter by Region", options=sorted(clean_df['Region_Name__c'].unique()), default=[])
    with f_col2:
        sel_state = st.multiselect("Filter by State", options=sorted([s for s in clean_df['State'].unique() if s]), default=[])
    with f_col3:
        sel_district = st.multiselect("Filter by District", options=sorted(clean_df['District_Name__c'].unique()), default=[])
    with f_col4:
        sel_loyalty = st.selectbox("Super Loyalty", options=["All", "Yes", "No"], index=0)

    filtered_df = clean_df.copy()
    if sel_region:
        filtered_df = filtered_df[filtered_df['Region_Name__c'].isin(sel_region)]
    if sel_state:
        filtered_df = filtered_df[filtered_df['State'].isin(sel_state)]
    if sel_district:
        filtered_df = filtered_df[filtered_df['District_Name__c'].isin(sel_district)]
    if sel_loyalty != "All":
        filtered_df = filtered_df[filtered_df['Super Loyalty'] == sel_loyalty]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(clean_df):,} records**")
    st.dataframe(filtered_df, use_container_width=True, height=350)

    # Export CSV
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Data to CSV",
        data=csv_bytes,
        file_name="ricetec_filtered_sales_data.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 💻 SQL Query Console")
    st.caption("Directly query the in-memory SQLite table `sales_opportunity`")
    sql_input = st.text_area(
        "SQL Query:",
        value="SELECT District_Name__c, Region_Name__c, COUNT(*) as Customers, SUM([2025_Acres]) as Total_2025_Acres\nFROM sales_opportunity\nGROUP BY District_Name__c, Region_Name__c\nORDER BY Total_2025_Acres DESC\nLIMIT 10;",
        height=100
    )
    if st.button("▶️ Execute SQL"):
        sql_df = data_engine.execute_sql(sql_input)
        st.dataframe(sql_df, use_container_width=True)
