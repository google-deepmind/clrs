"""Tests for clrs._src.clrs_text.huggingface_generators."""

from datasets import Dataset, IterableDataset, Value, Features

from absl.testing import parameterized
import clrs
from clrs._src.clrs_text import clrs_utils, huggingface_generators


class TestFormatCLRSExamplesHFDataset(parameterized.TestCase):
    """Based on TestFormatCLRSExamples in clrs.src_.clrs_text.clrs_utils_test.py"""

    @parameterized.product(
        algo_name=list(clrs.CLRS_30_ALGS_SETTINGS.keys()),
        lengths=[[16], [16, 32]],
        use_hints=[True, False],
    )
    def test_format(self, algo_name, lengths, use_hints):
        """Test that we can format samples from any algo into strings from a hf Dataset."""
        algos_and_lengths = {algo_name: lengths}
        ds = Dataset.from_generator(
            huggingface_generators.clrs_generator,
            gen_kwargs={
                "algos_and_lengths": algos_and_lengths,
                "num_samples": 100,
                "use_hints": use_hints,
            },
        )

        for sample in ds:
            (
                text,
                question,
                answer,
                sample_algo_name,
                sample_length,
                use_hints,
            ) = (
                sample["text"],
                sample["question"],
                sample["answer"],
                sample["algo_name"],
                sample["length"],
                sample["use_hints"],
            )

            self.assertEqual(algo_name, sample_algo_name)
            self.assertEqual(use_hints, use_hints)
            self.assertIn(sample_length, lengths)

            self.assertTrue(question.startswith(f"{algo_name}:\n"))
            self.assertTrue(question.endswith(":\n"))
            self.assertTrue(answer.endswith("\n\n"))

            self.assertTrue(text.startswith(f"{algo_name}:\n"))
            self.assertTrue(text.endswith("\n\n"))
            self.assertEqual(question + answer, text)

            if (
                use_hints and algo_name in clrs_utils.CLRS_TASKS_WITH_HINTS
            ):  # segments intersect has no hints option
                self.assertIn("trace | ", question)
                self.assertIn("initial_trace:", question)
                self.assertIn("trace | ", text)
                self.assertIn("initial_trace:", text)
            else:
                self.assertNotIn("trace | ", question)
                self.assertNotIn("initial_trace:", question)
                self.assertNotIn("trace | ", text)
                self.assertNotIn("initial_trace:", text)


class TestFormatCLRSExamplesHFIterableDataset(parameterized.TestCase):
    """Based on TestFormatCLRSExamples in clrs.src_.clrs_text.clrs_utils_test.py"""

    @parameterized.product(
        algo_name=list(clrs.CLRS_30_ALGS_SETTINGS.keys()),
        lengths=[[16], [16, 32]],
        use_hints=[True, False],
    )
    def test_format(self, algo_name, lengths, use_hints):
        """Test that we can format samples from any algo into strings from a hf IterableDataset."""
        algos_and_lengths = {algo_name: lengths}
        ds = IterableDataset.from_generator(
            huggingface_generators.clrs_infinite_generator,
            features=Features(
                {
                    "text": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "answer": Value(dtype="string", id=None),
                    "algo_name": Value(dtype="string", id=None),
                    "length": Value(dtype="int32", id=None),
                    "use_hints": Value(dtype="bool_", id=None),
                }
            ),
            gen_kwargs={"algos_and_lengths": algos_and_lengths, "use_hints": use_hints},
        )

        ds_iterator = iter(ds)
        for _ in range(100):  # only test 100 samples as we have infinite sampling on
            sample = next(ds_iterator)
            (
                text,
                question,
                answer,
                sample_algo_name,
                sample_length,
                use_hints,
            ) = (
                sample["text"],
                sample["question"],
                sample["answer"],
                sample["algo_name"],
                sample["length"],
                sample["use_hints"],
            )

            self.assertEqual(algo_name, sample_algo_name)
            self.assertEqual(use_hints, use_hints)
            self.assertIn(sample_length, lengths)

            self.assertTrue(question.startswith(f"{algo_name}:\n"))
            self.assertTrue(question.endswith(":\n"))
            self.assertTrue(answer.endswith("\n\n"))

            self.assertTrue(text.startswith(f"{algo_name}:\n"))
            self.assertTrue(text.endswith("\n\n"))
            self.assertEqual(question + answer, text)

            if (
                use_hints and algo_name in clrs_utils.CLRS_TASKS_WITH_HINTS
            ):  # segments intersect has no hints option
                self.assertIn("trace | ", question)
                self.assertIn("initial_trace:", question)
                self.assertIn("trace | ", text)
                self.assertIn("initial_trace:", text)
            else:
                self.assertNotIn("trace | ", question)
                self.assertNotIn("initial_trace:", question)
                self.assertNotIn("trace | ", text)
                self.assertNotIn("initial_trace:", text)
