from .voicebench.voicebench import VoiceBench
from .mmau.mmau import MMAU
from .openaudiobench.openaudiobench import OpenAudioBench
from .cmmlu import CMMLU

benchmark_mapping = {
    'voicebench': VoiceBench,
    'mmau': MMAU,
    'openaudiobench': OpenAudioBench,
    'cmmlu': CMMLU,
    # Add other benchmarks here as needed
}


def load_benchmark(benchmark_name, subset_name, split, **kwargs):
    """
    Load a dataset based on the provided benchmark name, subset name, and split.
    
    Args:
        benchmark_name (str): The name of the benchmark to load.
        subset_name (str): The name of the subset within the benchmark.
        split (str): The split of the benchmark to load (e.g., 'train', 'test').
    """
    if benchmark_name not in benchmark_mapping:
        raise ValueError(f"Benchmark {benchmark_name} is not supported.")
    
    dataset_class = benchmark_mapping[benchmark_name]
    return dataset_class(subset_name=subset_name, split=split, **kwargs)


def list_benchmarks():
    """
    List all available benchmarks.
    
    Returns:
        list: A list of available benchmark names.
    """
    return list(benchmark_mapping.keys())
