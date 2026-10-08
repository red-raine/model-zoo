"""assembler tests — plan assembly + dependency checks."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assembler import classify, plan  # noqa: E402


class TestAssembler(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("code coder model"), "code")
        self.assertEqual(classify("ternary 1.58 bit model"), "ternary")
        self.assertEqual(classify("distill to tiny"), "distill")
        self.assertEqual(classify("a generic model"), "default")

    def test_plan_default_no_warnings(self):
        zp = plan("build a small model from traces")
        self.assertTrue(zp.stages)
        self.assertEqual(zp.stages[0].name, "spec")
        self.assertTrue(zp.stages[-1].name in ("serve", "code_served"))

    def test_plan_code_chain(self):
        zp = plan("3b coding model distilled")
        names = [s.name for s in zp.stages]
        self.assertIn("train", names)
        self.assertIn("distill", names)
        self.assertIn("eval", names)

    def test_dependency_check(self):
        # every stage's input must be produced by an upstream stage (or a spec)
        for intent_name in ("default", "code"):
            zp = plan(f"a {intent_name} project")
            produced = {"spec.md", "spec_code.md"}
            for st in zp.stages:
                for i in st.inputs:
                    self.assertIn(i, produced, f"{st.name} needs {i}")
                produced.update(st.outputs)


if __name__ == "__main__":
    unittest.main()