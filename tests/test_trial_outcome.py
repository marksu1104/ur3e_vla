"""Offline checks for explicit, staged trial outcome evidence."""

from __future__ import annotations

import unittest

from vla_sim.pick_place import TRIAL_OUTCOME_FIELDS, classify_trial_outcome


class TrialOutcomeTest(unittest.TestCase):
    def all_stages(self, value: bool | None) -> dict[str, bool | None]:
        return {field: value for field in TRIAL_OUTCOME_FIELDS}

    def test_all_explicit_stages_pass(self) -> None:
        result = classify_trial_outcome(self.all_stages(True))
        self.assertEqual(result["pipeline"], "pass")
        self.assertEqual(result["trajectory"], "pass")
        self.assertEqual(result["physical_pick_place"], "pass")
        self.assertEqual(result["end_to_end"], "pass")

    def test_absent_stage_evidence_stays_unknown(self) -> None:
        result = classify_trial_outcome({})
        self.assertEqual(result["physical_pick_place"], "unknown")
        self.assertEqual(result["end_to_end"], "unknown")
        self.assertTrue(all(value == "unknown" for value in result["stages"].values()))

    def test_http_exit_and_complete_signals_do_not_prove_task_success(self) -> None:
        result = classify_trial_outcome(
            {"http_status": 200, "exit_code": 0, "complete": True}
        )
        self.assertEqual(result["pipeline"], "unknown")
        self.assertEqual(result["trajectory"], "unknown")
        self.assertEqual(result["physical_pick_place"], "unknown")
        self.assertEqual(result["end_to_end"], "unknown")

    def test_failed_physical_stage_fails_pick_place_and_end_to_end(self) -> None:
        evidence = self.all_stages(True)
        evidence["transported_without_drop"] = False
        result = classify_trial_outcome(evidence)
        self.assertEqual(result["physical_pick_place"], "fail")
        self.assertEqual(result["end_to_end"], "fail")
        self.assertEqual(result["stages"]["transported_without_drop"], "fail")

    def test_pipeline_failure_is_separate_from_physical_stage_summary(self) -> None:
        evidence = self.all_stages(True)
        evidence["pipeline_ok"] = False
        result = classify_trial_outcome(evidence)
        self.assertEqual(result["pipeline"], "fail")
        self.assertEqual(result["physical_pick_place"], "pass")
        self.assertEqual(result["end_to_end"], "fail")

    def test_trajectory_failure_is_reported_separately(self) -> None:
        evidence = self.all_stages(True)
        evidence["trajectory_complete"] = False
        result = classify_trial_outcome(evidence)
        self.assertEqual(result["trajectory"], "fail")
        self.assertEqual(result["physical_pick_place"], "pass")
        self.assertEqual(result["end_to_end"], "fail")

    def test_none_is_unknown_even_when_other_stages_pass(self) -> None:
        evidence = self.all_stages(True)
        evidence["released"] = None
        result = classify_trial_outcome(evidence)
        self.assertEqual(result["stages"]["released"], "unknown")
        self.assertEqual(result["physical_pick_place"], "unknown")
        self.assertEqual(result["end_to_end"], "unknown")

    def test_non_boolean_stage_values_are_rejected(self) -> None:
        for value in (0, 1, "true", "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "pipeline_ok"):
                    classify_trial_outcome({"pipeline_ok": value})

    def test_non_mapping_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping"):
            classify_trial_outcome(None)  # type: ignore[arg-type]

    def test_unrelated_evidence_fields_do_not_change_stage_classification(self) -> None:
        evidence = self.all_stages(None)
        evidence.update({"complete": True, "exit_code": 0, "http_status": 200})
        result = classify_trial_outcome(evidence)
        self.assertEqual(result["end_to_end"], "unknown")


if __name__ == "__main__":
    unittest.main()
