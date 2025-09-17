benchmark_mapping = {
    'voicebench': ('.voicebench.voicebench', 'VoiceBench'),
    'mmau': ('.mmau.mmau', 'MMAU'),
    'openaudiobench': ('.openaudiobench.openaudiobench', 'OpenAudioBench'),
    'cmmlu_gen': ('.cmmlu_gen', 'CMMLU'),
    'cmmlu_ppl': ('.cmmlu_ppl', 'CMMLU'),
    'mmlu_ppl': ('.mmlu_ppl', 'MMLU'),
    'mmlu_gen': ('.mmlu_gen', 'MMLU'),
    'zh-storycloze': ('.zh_storycloze', 'zhStoryCloze'),
    'voxeval': ('.voxeval', 'VoxEval'),
    'storycloze': ('.storycloze', 'StoryCloze'),
    'storycloze_test': ('.storycloze_test', 'StoryCloze'),
    'airbench': ('.airbench.airbench', 'AIRBench'),
    'seedttseval': ('.seed_tts_eval.seed_tts_eval', 'SeedTTSEval'),
    'commonvoice': ('.commonvoice', 'CommonVoice'),
    'mmar': ('.mmar.mmar', 'MMAR'),
    'fullduplexbench': ('.fullduplexbench','FullDuplexBench'),
    'spokenqa': ('.spokenqa.spokenqa', 'SpokenQA'),
    # pure text benchmark as follows
    '':('.',''),
}


def load_benchmark(benchmark, **kwargs):
    """
    Load a dataset based on the provided benchmark name, subset name, and split.
    
    Args:
        benchmark (str): The name of the benchmark to load.
    """
    if benchmark not in benchmark_mapping:
        raise ValueError(f"Benchmark {benchmark} is not supported.")
    
    import importlib
    module_path, class_name = benchmark_mapping[benchmark]
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
