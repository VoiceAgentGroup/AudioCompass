benchmark_mapping = {
    'voicebench': ('.voicebench.voicebench', 'VoiceBench'),
    'mmau': ('.mmau.mmau', 'MMAU'),
    'openaudiobench': ('.openaudiobench.openaudiobench', 'OpenAudioBench'),
    'cmmlu': ('.cmmlu', 'CMMLU'),
    'mmlu': ('.mmlu', 'MMLU'),
    'zh-storycloze': ('.zh_storycloze', 'zhStoryCloze'),
    'voxeval': ('.voxeval', 'VoxEval'),
    'storycloze': ('.storycloze', 'StoryCloze'),
    'airbench': ('.airbench.airbench', 'AIRBench'),
    'seedttseval': ('.seed_tts_eval.seed_tts_eval', 'SeedTTSEval')
}


def load_benchmark(benchmark_name, **kwargs):
    """
    Load a dataset based on the provided benchmark name, subset name, and split.
    
    Args:
        benchmark_name (str): The name of the benchmark to load.
        subset_name (str): The name of the subset within the benchmark.
        split (str): The split of the benchmark to load (e.g., 'train', 'test').
    """
    if benchmark_name not in benchmark_mapping:
        raise ValueError(f"Benchmark {benchmark_name} is not supported.")
    
    import importlib
    module_path, class_name = benchmark_mapping[benchmark_name]
    module = importlib.import_module(module_path, package="src.benchmarks")
    dataset_class = getattr(module, class_name)
    
    return dataset_class(**kwargs)


def list_benchmarks():
    """
    List all available benchmarks.
    
    Returns:
        list: A list of available benchmark names.
    """
    return list(benchmark_mapping.keys())
