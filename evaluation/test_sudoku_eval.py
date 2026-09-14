import unittest

from evaluate_sudoku import (
    DIRECT_SYSTEM_PROMPT,
    RULES_SYSTEM_PROMPT,
    prompt_messages,
    score_prediction,
    valid_solution,
    wilson_interval,
)


PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


class SudokuEvaluationTests(unittest.TestCase):
    def test_rules_prompt_explicitly_states_all_constraints(self):
        self.assertIn("Every row must contain each digit 1 through 9", RULES_SYSTEM_PROMPT)
        self.assertIn("Every column must contain each digit 1 through 9", RULES_SYSTEM_PROMPT)
        self.assertIn("Every 3x3 box must contain each digit 1 through 9", RULES_SYSTEM_PROMPT)
        self.assertIn("Digits already given", RULES_SYSTEM_PROMPT)
        self.assertNotEqual(DIRECT_SYSTEM_PROMPT, RULES_SYSTEM_PROMPT)

    def test_prompt_messages_use_the_selected_system_prompt(self):
        messages = prompt_messages(PUZZLE, RULES_SYSTEM_PROMPT)
        self.assertEqual(messages[0]["content"], RULES_SYSTEM_PROMPT)
        self.assertIn("530070000", messages[1]["content"])

    def test_valid_solution_checks_rows_columns_and_boxes(self):
        self.assertTrue(valid_solution(SOLUTION))
        self.assertFalse(valid_solution(SOLUTION[:80] + "1"))

    def test_scoring_normalizes_whitespace_without_extracting_text(self):
        generated = "\n".join(SOLUTION[index : index + 9] for index in range(0, 81, 9))
        result = score_prediction(PUZZLE, SOLUTION, generated)
        self.assertTrue(result["parseable"])
        self.assertTrue(result["valid_with_clues"])
        self.assertTrue(result["exact_solution"])
        self.assertFalse(result["strict_exact_solution"])
        self.assertEqual(result["correct_blank_cells"], PUZZLE.count("0"))

    def test_explanations_are_not_repaired(self):
        result = score_prediction(PUZZLE, SOLUTION, f"Answer: {SOLUTION}")
        self.assertFalse(result["parseable"])
        self.assertFalse(result["exact_solution"])

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(5, 10)
        self.assertLessEqual(lower, 0.5)
        self.assertGreaterEqual(upper, 0.5)


if __name__ == "__main__":
    unittest.main()
