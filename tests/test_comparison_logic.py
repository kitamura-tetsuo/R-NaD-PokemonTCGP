import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Add root to sys.path
sys.path.append(os.getcwd())

# Mock streamlit before importing src.pages.comparison
with patch.dict(sys.modules, {'streamlit': MagicMock()}):
    from src.pages.comparison import parse_winrates, load_winrates

class TestComparisonLogic(unittest.TestCase):
    def test_parse_winrates_simple(self):
        data = {
            "600_mewtwoex_vs_mewtwoex_vs_control_500": {
                "p1_wins": 5,
                "total": 10
            },
            "700_mewtwoex_vs_mewtwoex_vs_control_500": {
                "p1_wins": 8,
                "total": 10
            }
        }

        result = parse_winrates(data)

        self.assertIn("mewtwoex_vs_mewtwoex", result)
        entry = result["mewtwoex_vs_mewtwoex"]
        self.assertEqual(entry["steps"], [600, 700])
        self.assertEqual(entry["win_rates"], [0.5, 0.8])

    def test_parse_winrates_mixed_keys(self):
        data = {
            "600_A_vs_B_vs_control_500": {"p1_wins": 1, "total": 2},
            "invalid_key": {"p1_wins": 0, "total": 0},
            "700_C_vs_D_vs_control_500": {"p1_wins": 2, "total": 2}
        }

        result = parse_winrates(data)

        self.assertIn("A_vs_B", result)
        self.assertEqual(result["A_vs_B"]["steps"], [600])
        self.assertEqual(result["A_vs_B"]["win_rates"], [0.5])

        self.assertIn("C_vs_D", result)
        self.assertEqual(result["C_vs_D"]["steps"], [700])
        self.assertEqual(result["C_vs_D"]["win_rates"], [1.0])

        self.assertNotIn("invalid_key", result) # Should be ignored (parsed as pair key "invalid_key"?) No, regex requires digits at start/end.

    def test_sorting(self):
        data = {
            "800_A_vs_B_vs_control_500": {"p1_wins": 3, "total": 4}, # 0.75
            "600_A_vs_B_vs_control_500": {"p1_wins": 1, "total": 4}  # 0.25
        }

        result = parse_winrates(data)

        self.assertIn("A_vs_B", result)
        self.assertEqual(result["A_vs_B"]["steps"], [600, 800])
        self.assertEqual(result["A_vs_B"]["win_rates"], [0.25, 0.75])

if __name__ == '__main__':
    unittest.main()
