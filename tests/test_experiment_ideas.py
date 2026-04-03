import unittest

from experiment_idea_library import expand_template_candidates, list_idea_templates, load_external_idea_seeds


class ExperimentIdeaLibraryTests(unittest.TestCase):
    def test_template_catalog_contains_broader_sources(self):
        templates = list_idea_templates("momentum")
        source_types = {template.source_type for template in templates}
        self.assertIn("template_expansion", source_types)
        self.assertIn("cross_family_hybrid", source_types)
        self.assertIn("model_based", {template.source_type for template in list_idea_templates("ml_ranker")})
        self.assertIn("policy_learning", {template.source_type for template in list_idea_templates("rl_bandit")})
        self.assertIn("ml", {template.strategy_type for template in list_idea_templates("ml_ranker")})
        self.assertIn("rl", {template.strategy_type for template in list_idea_templates("rl_bandit")})

    def test_template_payloads_are_normalized_and_tagged(self):
        payloads = expand_template_candidates("superstock", limit=5, seed=7)
        self.assertEqual(len(payloads), 5)
        for payload in payloads:
            self.assertIn("config", payload)
            self.assertIn("metadata", payload)
            self.assertLessEqual(payload["config"]["price_min"], payload["config"]["price_max"])
            self.assertIn(payload["metadata"]["source_type"], {"template_expansion", "cross_family_hybrid"})
            self.assertEqual(payload["metadata"]["strategy_type"], "classical")

    def test_external_seed_adapter_defaults_to_empty(self):
        self.assertEqual(load_external_idea_seeds(enabled=False), [])


if __name__ == "__main__":
    unittest.main()
