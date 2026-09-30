"""
test_chatbot.py - Automated tests for RiceTec Sales & Opportunity Chatbot Engine.
Tests dynamic calculation, step-by-step methodology, and future-ready evaluation across all categories.
"""

import unittest
import pandas as pd
from data_engine import SalesDataEngine
from query_engine import QueryEngine
import knowledge_bank

class TestSalesChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_engine = SalesDataEngine()
        cls.query_engine = QueryEngine(cls.data_engine)

    def test_kpi_calculations(self):
        k = self.data_engine.kpis
        self.assertEqual(k['total_records'], 2415)
        self.assertEqual(k['unique_customers'], 2354)
        self.assertGreater(k['total_2025_acres'], 1_000_000)
        self.assertGreater(k['total_oppor'], 1_500_000)
        # Sum of technologies equals 2025 total acres
        tech_sum = k['total_full_page_2025'] + k['total_max_ace_2025'] + k['total_non_ht_2025'] + k['total_non_ht_mg_2025']
        self.assertAlmostEqual(tech_sum, k['total_2025_acres'], places=1)

    def test_fuzzy_matching(self):
        # Exact name
        res = self.data_engine.fuzzy_match_customer("DONNY DELINE")
        self.assertTrue(len(res) > 0)
        self.assertIn("DONNY DELINE", res[0])

        # Partial/misspelled name
        res2 = self.data_engine.fuzzy_match_customer("donny delin")
        self.assertTrue(len(res2) > 0)
        self.assertIn("DONNY DELINE", res2[0])

        # ID lookup
        res3 = self.data_engine.fuzzy_match_customer("1000006621")
        self.assertEqual(len(res3), 1)
        self.assertIn("1000006621", res3[0])

    def test_customer_profile(self):
        profile = self.data_engine.get_customer_profile("DONNY DELINE - 1000006621")
        self.assertIsNotNone(profile)
        self.assertEqual(profile['rebates']['super_loyalty'], 'Yes')
        self.assertGreater(profile['total_2025_acres'], 0)
        self.assertGreater(profile['total_oppor'], 0)

    def test_multi_county_customer(self):
        # 4 J FARMS has multiple locations
        matches = self.data_engine.fuzzy_match_customer("4 J FARMS")
        self.assertTrue(len(matches) > 0)
        profile = self.data_engine.get_customer_profile(matches[0])
        self.assertGreaterEqual(profile['row_count'], 2)
        self.assertGreaterEqual(len(profile['locations']), 2)

    def test_queries(self):
        queries = [
            "Hi",
            "Executive summary",
            "Tell me about Craig Zaunbrecher",
            "Top 10 customers by 2025 acres",
            "Top 5 districts by opportunity",
            "What is the product mix for 2025?",
            "How many Super Loyalty customers?",
            "Summarize Region R01",
            "District D11",
            "How many acres in Louisiana?",
            "Compare 2024 vs 2025 acres",
            "Who has the biggest opportunity gap?",
            "give me 2025 Seed Technology Breakdown for DONNY DELINE - 1000006621"
        ]
        # Specifically verify customer seed tech breakdown is customer-specific
        res_tech = self.query_engine.process_query("give me 2025 Seed Technology Breakdown for DONNY DELINE - 1000006621")
        self.assertIn("DONNY DELINE", res_tech["text"])
        self.assertIn("Full Page", res_tech["text"])
        self.assertIn("5,393.2", res_tech["text"])
        for q in queries:
            res = self.query_engine.process_query(q)
            self.assertIsNotNone(res)
            self.assertIn("text", res)
            self.assertTrue(len(res["text"]) > 10, f"Query '{q}' returned too short text: {res['text']}")

    def test_benchmark_step_by_step_and_dynamic_rules(self):
        # Category 1: Portfolio (ID 1)
        r1 = self.query_engine.process_query("How many customer records are in the dataset?")
        self.assertIn("2415 customers", r1["text"])
        self.assertIn("Step-by-Step Calculation Methodology", r1["text"])
        self.assertIn("Reusable Dynamic Rule", r1["text"])

        # Category 1: Total Acres 2023 (ID 2)
        r2 = self.query_engine.process_query("What are total acres in 2023?")
        self.assertIn("1,396,773.78 acres", r2["text"])

        # Category 1: Weighted Growth 2024-2025 (ID 6)
        r6 = self.query_engine.process_query("What was weighted growth from 2024 to 2025?")
        self.assertIn("-25.56%", r6["text"])
        self.assertIn("-354,319.45", r6["text"])

        # Category 3: District D07 (ID 39)
        rd = self.query_engine.process_query("What are total 2025 acres in district D07?")
        self.assertIn("111,582.80 acres across 178 customers", rd["text"])

        # Category 4: State AR Growth (ID 88)
        rst = self.query_engine.process_query("What was weighted 2025 growth in AR?")
        self.assertIn("-27.63%", rst["text"])

        # Category 5: Customer Ranking #1 (ID 114)
        rc = self.query_engine.process_query("Who ranks #1 by 2025 acres?")
        self.assertIn("DONNY DELINE", rc["text"])
        self.assertIn("8,073.20 acres", rc["text"])

        # Category 7: Product Mix Full Page (ID 139)
        rpm = self.query_engine.process_query("What are total 2025 Full Page acres?")
        self.assertIn("678,819.92 acres", rpm["text"])
        self.assertIn("65.78%", rpm["text"])

        # Category 8: Data Quality Reconciliation (ID 143)
        rdq = self.query_engine.process_query("Do the four product categories reconcile to 2025 acres for every customer?")
        self.assertIn("Yes. All 2415 rows reconcile within 0.01 acre.", rdq["text"])

        # Category 10: Top 10 Concentration (ID 156)
        rbi = self.query_engine.process_query("What share of 2025 acres is held by the top 10 customers?")
        self.assertIn("5.78%", rbi["text"])
        self.assertIn("59,660.48 acres", rbi["text"])

    def test_multi_condition_rebate_filters(self):
        # Query 1: Loyalty=Yes, Volume=Yes, Super Loyalty=No (ID 150 -> 90 customers, 106,983.53 acres)
        res1 = self.query_engine.process_query("How many customers have Loyalty=Yes and Volume=Yes but Super Loyalty=No?")
        self.assertIn("90 customers", res1["text"])
        self.assertIn("106,983.53", res1["text"])
        self.assertIn("Step-by-Step Calculation Methodology", res1["text"])
        self.assertNotIn("YES DEERE FARMS", res1["text"])

        # Query 2: Super Loyalty=Yes, Volume=No (ID 151 -> 107 customers, 30,361.73 acres)
        res2 = self.query_engine.process_query("How many have Super Loyalty=Yes but Volume Rebate=No?")
        self.assertIn("107 customers", res2["text"])
        self.assertIn("30,361.73", res2["text"])
        self.assertNotIn("YES DEERE FARMS", res2["text"])

        # Query 3: Super Loyalty=Blank (ID 146 -> 284 customers, 21,561.31 acres)
        res3 = self.query_engine.process_query("How many customers have Super Loyalty=Blank?")
        self.assertIn("284 customers", res3["text"])
        self.assertIn("21,561.31", res3["text"])

    def test_consecutive_growth(self):
        # Consecutive growth 2023 < 2024 < 2025 (ID 16 -> 136 customers)
        res = self.query_engine.process_query("How many customers grew in both 2024 and 2025 compared with the preceding year?")
        self.assertIn("136 customers", res["text"])
        self.assertIn("Step-by-Step Calculation Methodology", res["text"])
        self.assertIn("Reusable Dynamic Rule", res["text"])

    def test_future_proof_dynamic_recalculation(self):
        # Verify that evaluating on modified data updates numbers dynamically
        base_df = self.data_engine.clean_df
        orig_cust = knowledge_bank.evaluate_benchmark(1, base_df)
        self.assertEqual(orig_cust["computed_answer"], "2415 customers.")

        # Simulate addition of new customer rows
        sim_row = base_df.iloc[0:1].copy()
        sim_row["Customer_Group__c"] = "FUTURE NEW FARM - 9999999999"
        sim_row["2025 Acres"] = 10000.0
        sim_df = pd.concat([base_df, sim_row], ignore_index=True)

        sim_cust = knowledge_bank.evaluate_benchmark(1, sim_df)
        self.assertEqual(sim_cust["computed_answer"], "2416 customers.")

        sim_acres = knowledge_bank.evaluate_benchmark(4, sim_df)
        self.assertEqual(sim_acres["computed_answer"], "1,041,945.68 acres.")

if __name__ == "__main__":
    unittest.main()
