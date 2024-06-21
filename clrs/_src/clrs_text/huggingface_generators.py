"""Functions to allow for Huggingface Intergration"""
from typing import List, Dict
from clrs import build_sampler
from clrs._src.clrs_text.clrs_utils import format_clrs_example
import itertools


def clrs_gen(algs: Dict[str, List[int]], num_samples: int, seed: int = 0, use_hints: bool = True, pretrain: bool = False):
    """
    Huggingface datasets.Dataset compatible generator function for creating a dataset of fixed size

    Example usage:
        import datasets
        algs = {"insertion_sort": [16]}
        ds = datasets.Dataset.from_generator(clrs_gen, gen_kwargs={"algs": algs, "num_samples": 100})

    Huggingface reference:
        https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.from_generator

    Args:
        algs (Dict[str, List[int]]): keys = algorithm names [Must be same as in clrs.CLRS_30_ALGS_SETTINGS.keys()], values = list of lengths required for that algorithm
        num_samples (int): The size of the output dataset
        seed (int), default=0: The random seed for all of the generators
        use_hints (bool), default=True: Whether to include hints in the questions
        pretrain (bool), default=False: Whether to combine the question and answer into one field called text
 
    Returns:
        Sample question and answer as a dictionary
    """
    choices = {}

    # make all of the possible generators
    for alg in algs.keys():
        for length in algs[alg]:
            sampler, _ = build_sampler(
                alg,
                seed=seed,
                num_samples=-1,
                length=length,
                track_max_steps=False,
                use_padding=False,
            )
            choices[f"{alg}={length}"] = sampler

    keys = itertools.cycle(list(choices.keys())) # make a cyclic list of all the generators

    # uniformly sample one element from each sampler in the list up to the maximum
    for _ in range(num_samples):
        sampler_name = next(keys)
        sample = choices[sampler_name].next(batch_size=1) # get one sample from the sampler
        algorithm_name = sampler_name.split('=')[0] # algorithm name, split at = sign
        algorithm_size = sampler_name.split('=')[1]
        question, answer = format_clrs_example(
                                algorithm_name,
                                sample,
                                use_hints=use_hints,
                            )
        if pretrain:
            text = question + answer
            yield {"text": text, "name": algorithm_name, "size": algorithm_size}
        else:
            yield {"question": question, "answer": answer, "name": algorithm_name, "size": algorithm_size}


def clrs_gen_inf(algs: Dict[str, List[int]], seed: int = 0, use_hints: bool = True, pretrain: bool = False):
    """
    Huggingface datasets.IterableDataset compatible generator function for creating a dataset of infinite size

    Example usage:
        import datasets
        algs = {"insertion_sort": [16]}
        ds = datasets.IterableDataset.from_generator(clrs_gen_inf, features=datasets.Features({'question': datasets.Value(dtype='string', id=None), 'answer': datasets.Value(dtype='string', id=None)}), gen_kwargs={"algs": algs}
        ds = IterableDataset.from_generator(clrs_gen_inf, features=Features({'question': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None), 'name': Value(dtype='string', id=None), 'size': Value(dtype='string', id=None)}), gen_kwargs={"algs": alg_lens})
        
        OR if using pretraining = True:
        ds = IterableDataset.from_generator(clrs_gen_inf, features=Features({'text': Value(dtype='string', id=None), 'name': Value(dtype='string', id=None), 'size': Value(dtype='string', id=None)}), gen_kwargs={"algs": alg, "pretrain": True})
        
    Huggingface reference:
        https://huggingface.co/docs/datasets/v2.7.0/en/package_reference/main_classes#datasets.IterableDataset.from_generator

    Args:
        algs (Dict[str, List[int]]): keys = algorithm names [Must be same as in clrs.CLRS_30_ALGS_SETTINGS.keys()], values = list of lengths required for that algorithm
        seed (int), default=0: The random seed for all of the generators
        use_hints (bool), default=True: Whether to include hints in the questions
        pretrain (bool), default=False: Whether to combine the question and answer into one field called text
 
    Returns:
        Sample question and answer as a dictionary
    """
    choices = {}
    
    # make all of the possible generators
    for alg in algs.keys():
        for length in algs[alg]:
            sampler, _ = build_sampler(
                alg,
                seed=seed,
                num_samples=-1,
                length=length,
                track_max_steps=False,
                use_padding=False,
            )
            choices[f"{alg}={length}"] = sampler

    keys = itertools.cycle(list(choices.keys())) # make a cyclic list of all the generators

    # uniformly sample one element from each sampler in the list continuously
    while True:
        sampler_name = next(keys)
        sample = choices[sampler_name].next(batch_size=1) # get one sample from the sampler
        algorithm_name = sampler_name.split('=')[0] # algorithm name, split at = sign
        algorithm_size = sampler_name.split('=')[1]
        question, answer = format_clrs_example(
                                algorithm_name,
                                sample,
                                use_hints=use_hints,
                            )
        if pretrain:
            text = question + answer
            yield {"text": text, "name": algorithm_name, "size": algorithm_size}
        else:
            yield {"question": question, "answer": answer, "name": algorithm_name, "size": algorithm_size}