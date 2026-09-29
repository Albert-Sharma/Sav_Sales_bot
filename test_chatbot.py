"""
test_chatbot.py - Automated tests for RiceTec Sales & Opportunity Chatbot Engine.
"""

import unittest
from data_engine import SalesDataEngine
from query_engine import QueryEngine

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
            print(f"PASS: '{q}' -> Response length {len(res['text'])}")

if __name__ == "__main__":
    unittest.main()
