import unittest

from experiment_novelty import coarse_signature_key, is_near_duplicate, score_candidate
from experiment_store import compute_config_hash


class ExperimentNoveltyTests(unittest.TestCase):
    def test_exact_duplicate_blocking(self):
        config = {
            "LOOKBACK_WEEKS": 26,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20,
            "STOP_PARABOLIC": 0.30,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
        explored = {compute_config_hash("momentum", config)}
        novelty = score_candidate(
            "momentum",
            config,
            history=[],
            explored_hashes=explored,
            explored_signatures={},
            novelty_floor=0.0,
        )
        self.assertIsNone(novelty)

    def test_novelty_records_duplicate_risk_for_near_duplicates(self):
        config = {
            "LOOKBACK_WEEKS": 26,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20,
            "STOP_PARABOLIC": 0.30,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
        novelty = score_candidate(
            "momentum",
            config,
            history=[],
            explored_hashes=set(),
            explored_signatures={"other": coarse_signature_key("momentum", config)},
            novelty_floor=0.0,
        )
        self.assertIsNotNone(novelty)
        self.assertEqual(novelty.duplicate_risk, "near")

    def test_near_duplicate_detection(self):
        base = {
            "LOOKBACK_WEEKS": 26,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20,
            "STOP_PARABOLIC": 0.30,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
        other = dict(base)
        other["SKIP_WEEKS"] = 4
        explored = {compute_config_hash("momentum", base): coarse_signature_key("momentum", base)}
        near, near_of = is_near_duplicate("momentum", other, explored, max_distance=1)
        self.assertTrue(near)
        self.assertIsNotNone(near_of)

    def test_dead_zone_penalty_does_not_collapse_search(self):
        config = {
            "LOOKBACK_WEEKS": 20,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20,
            "STOP_PARABOLIC": 0.30,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
        novelty = score_candidate(
            "momentum",
            config,
            history=[],
            explored_hashes=set(),
            explored_signatures={},
            dead_zone_values={"LOOKBACK_WEEKS": {20}},
            novelty_floor=0.0,
        )
        self.assertIsNotNone(novelty)
        self.assertGreater(novelty.dead_zone_risk, 0.0)
        self.assertIn("LOOKBACK_WEEKS=20", novelty.dead_zone_flags)

    def test_saturation_escape_can_score_near_duplicate_without_exact_duplicate(self):
        base = {
            "LOOKBACK_WEEKS": 26,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20,
            "STOP_PARABOLIC": 0.30,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
        candidate = dict(base)
        candidate["SKIP_WEEKS"] = 4
        novelty = score_candidate(
            "momentum",
            candidate,
            history=[],
            explored_hashes={compute_config_hash("momentum", base)},
            explored_signatures={"base": coarse_signature_key("momentum", base)},
            source_type="saturation_escape",
            exploration_mode="saturation_escape",
            novelty_floor=0.15,
        )
        self.assertIsNotNone(novelty)
        self.assertEqual(novelty.duplicate_risk, "near")


if __name__ == "__main__":
    unittest.main()
