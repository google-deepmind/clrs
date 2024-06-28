"""Functions to allow for Huggingface Intergration"""

from typing import List, Dict
import random

from clrs import build_sampler
from clrs._src.clrs_text.clrs_utils import format_clrs_example


def clrs_generator(
    algos_and_lengths: Dict[str, List[int]],
    num_samples: int,
    use_hints: bool = False,
    seed: int = 0,
):
    """
    Huggingface datasets.Dataset generator function for creating a dataset of fixed size

    Example usage:
        import datasets
        algos_and_lengths = {"insertion_sort": [16]}
        ds = datasets.Dataset.from_generator(
            clrs_gen, gen_kwargs={"algos_and_lengths": algos_and_lengths, "num_samples": 100}
        )

    Huggingface reference:
        https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.from_generator

    Args:
        algos_and_lengths: keys = algorithm names
            [Must be same as in clrs.CLRS_30_ALGS_SETTINGS.keys()],
            values = list of lengths required for that algorithm
        num_samples: The size of the output dataset
        use_hints: Whether hints should be included in the question and answer
        seed: The random seed for all of the generators

    Returns:
        Sample question and answer in various formats with meta data as a dictionary
    """
    clrs_samplers = []

    # make all of the possible generators
    for algo_name, lengths in algos_and_lengths.items():
        for length in lengths:
            sampler, _ = build_sampler(
                algo_name,
                seed=seed,
                num_samples=-1,
                length=length,
                track_max_steps=False,
                use_padding=False,
            )
            clrs_samplers.append((sampler, algo_name, length))

    random.seed(seed)
    # uniformly sample one element from each sampler in the list up to the maximum
    for _ in range(num_samples):
        sampler, algo_name, length = random.choice(clrs_samplers)
        sample = sampler.next(batch_size=1)  # get one sample from the sampler
        question, answer = format_clrs_example(
            algo_name,
            sample,
            use_hints=use_hints,
        )

        text = question + answer
        yield {
            "text": text,
            "question": question,
            "answer": answer,
            "algo_name": algo_name,
            "length": length,
            "use_hints": use_hints,
        }


def clrs_infinite_generator(
    algos_and_lengths: Dict[str, List[int]], use_hints: bool = False, seed: int = 0
):
    """
    Huggingface datasets.Dataset generator function for creating a dataset of fixed size

    Example usage:
        import datasets
        algos_and_lengths = {"insertion_sort": [16]}
        ds = IterableDataset.from_generator(
            clrs_infinite_generator,
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
            gen_kwargs={"algos_and_lengths": algos_and_lengths},
        )

    Huggingface reference:
        https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.from_generator

    Args:
        algos_and_lengths: keys = algorithm names
            [Must be same as in clrs.CLRS_30_ALGS_SETTINGS.keys()],
            values = list of lengths required for that algorithm
        seed: The random seed for all of the generators

    Returns:
        Sample question and answer in various formats with meta data as a dictionary
    """
    clrs_samplers = []

    # make all of the possible generators
    for algo_name, lengths in algos_and_lengths.items():
        for length in lengths:
            sampler, _ = build_sampler(
                algo_name,
                seed=seed,
                num_samples=-1,
                length=length,
                track_max_steps=False,
                use_padding=False,
            )
            clrs_samplers.append((sampler, algo_name, length))

    random.seed(seed)
    # uniformly sample one element from each sampler in the list up to the maximum
    while True:
        sampler, algo_name, length = random.choice(clrs_samplers)
        sample = sampler.next(batch_size=1)  # get one sample from the sampler
        question, answer = format_clrs_example(
            algo_name,
            sample,
            use_hints=use_hints,
        )

        text = question + answer
        yield {
            "text": text,
            "question": question,
            "answer": answer,
            "algo_name": algo_name,
            "length": length,
            "use_hints": use_hints,
        }
