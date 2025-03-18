from .voicebench.voicebench import VoiceBench

benchmark_mapping = {
    'voicebench': VoiceBench,
}

def load_benchmark(benchmark_name, subset_name, split):
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
    return dataset_class(subset_name=subset_name, split=split)


def list_benchmarks():
    """
    List all available benchmarks.
    
    Returns:
        list: A list of available benchmark names.
    """
    return list(benchmark_mapping.keys())
