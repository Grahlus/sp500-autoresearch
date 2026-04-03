import unittest

from experiment_spaces import (
    get_family_default_config,
    get_family_search_space,
    list_searchable_families,
    normalize_experiment_config,
    validate_experiment_config,
)


class ExperimentSpacesTests(unittest.TestCase):
    def test_list_searchable_families(self):
        self.assertIn("momentum", list_searchable_families())
        self.assertIn("superstock", list_searchable_families())

    def test_default_config_matches_current_momentum_defaults(self):
        defaults = get_family_default_config("momentum")
        self.assertEqual(defaults["LOOKBACK_WEEKS"], 26)
        self.assertEqual(defaults["SKIP_WEEKS"], 3)
        self.assertEqual(defaults["STOP_TYPE"], "adaptive")
        self.assertEqual(defaults["STOP_LOSS_PCT"], 0.20)
        self.assertEqual(defaults["EXIT_PCT_RANK"], 0.97)

    def test_default_config_matches_current_superstock_defaults(self):
        defaults = get_family_default_config("superstock")
        self.assertEqual(defaults["max_positions"], 5)
        self.assertEqual(defaults["price_min"], 5.0)
        self.assertEqual(defaults["price_max"], 15.0)
        self.assertEqual(defaults["rs_rank_26w_min"], 0.70)

    def test_momentum_conditionals_are_normalized(self):
        config = normalize_experiment_config(
            "momentum",
            {
                "STOP_TYPE": "none",
                "STOP_LOSS_PCT": 0.20,
                "STOP_PARABOLIC": 0.30,
                "EXIT_PCT_RANK": None,
                "RANK_EXIT_CONFIRM": 2,
            },
        )
        self.assertIsNone(config["STOP_LOSS_PCT"])
        self.assertIsNone(config["STOP_PARABOLIC"])
        self.assertIsNone(config["RANK_EXIT_CONFIRM"])

    def test_superstock_price_band_is_normalized(self):
        config = normalize_experiment_config(
            "superstock",
            {
                "price_min": 20.0,
                "price_max": 5.0,
            },
        )
        self.assertEqual(config["price_min"], 5.0)
        self.assertEqual(config["price_max"], 20.0)

    def test_unknown_key_is_invalid(self):
        valid, message = validate_experiment_config("momentum", {"NOT_A_PARAM": 1})
        self.assertFalse(valid)
        self.assertIn("Unknown config key", message)

    def test_metadata_rich_space_has_expected_fields(self):
        spec = get_family_search_space("superstock")["max_positions"]
        self.assertEqual(spec["name"], "max_positions")
        self.assertEqual(spec["type"], "int")
        self.assertIn("default", spec)
        self.assertIn("choices", spec)


if __name__ == "__main__":
    unittest.main()
