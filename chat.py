"""
chat.py - Interactive Command-Line Chatbot for RiceTec Sales & Opportunity dataset.
Run this directly in any terminal: python chat.py
"""

import sys
from data_engine import SalesDataEngine
from query_engine import QueryEngine

def main():
    print("=" * 65)
    print(" 🌾 RiceTec Sales & Opportunity Intelligence Chatbot (Offline)")
    print("=" * 65)
    print("Loading sales dataset (RiceTec Sales  Opportunity Navpreet.xlsx)...")
    
    try:
        data_engine = SalesDataEngine()
        query_engine = QueryEngine(data_engine)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return

    k = data_engine.kpis
    print(f" Ready! Loaded {k['total_records']:,} records across {k['unique_customers']:,} customers.")
    print("\nYou can type any question, for example:")
    print("  • 'Tell me about DONNY DELINE'")
    print("  • 'Top 10 customers by 2025 acres'")
    print("  • 'What is the 2025 product mix?'")
    print("  • 'Super Loyalty customers'")
    print("  • 'Summarize Region R01'")
    print("  • 'Type 'exit' or 'quit' to close.\n")
    print("-" * 65)

    while True:
        try:
            user_input = input("\n👉 Enter your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        res = query_engine.process_query(user_input)
        
        # Display response
        print("\n" + "=" * 50)
        print(res.get("text", "No response."))
        print("=" * 50)

        # If table is present, display top rows cleanly
        table = res.get("table")
        if table is not None and hasattr(table, 'empty') and not table.empty:
            print("\n📊 Data Summary:")
            print(table.to_string(index=False))

if __name__ == "__main__":
    main()
