# AudioCompass

AudioCompass is a comprehensive evaluation framework for audio and multimodal language models. This platform allows researchers and developers to benchmark various voice assistant models using standardized datasets and metrics.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Preparation](#model-preparation)
- [Usage and Evaluation](#usage-and-evaluation)
- [Adding New Models](#adding-new-models)
- [Adding New Benchmarks](#adding-new-benchmarks)

## Overview

AudioCompass provides a unified interface to evaluate the capabilities of various voice assistant models. The framework supports:
- Audio-only input processing
- Text-only input processing
- Mixed audio and text inputs
- Integration with various benchmarks and evaluation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/VoiceAgentGroup/AudioCompass.git
cd AudioCompass

# Set up the environment
conda create -n audiocompass python=3.12 -y
conda activate audiocompass
pip install -r requirements.txt
```

## Usage and Evaluation

To run a benchmark on a specific model:

```bash
python main.py --model <model_name> --benchmark <dataset_name> --subset <subset_name> --split <split_name> --output-dir <output_directory> --cache-dir <cache-directory>
```

Add the `--offline` flag to run in environments without internet access, assuming models and datasets are already cached:

For example:

```bash
python main.py --model speechgpt2 --benchmark voicebench --subset alpacaeval --split test --output-dir output --cache-dir cache --offline
```

To list available models and benchmarks:

```python
from src.models import list_models
print(list_models())

from src.benchmarks import list_benchmarks
print(list_benchmarks())
```

## Data Preparation

1. Benchmark datasets are expected to be located within a `datas` subdirectory inside the specified cache directory (`--cache-dir`, defaults to `./cache`).
2. Ensure the cache directory exists (e.g., create `./cache/datas`).
3. For datasets requiring manual download, place the datasets into `<cache_dir>/datas/`:
    - [OpenAudioBench](https://huggingface.co/datasets/baichuan-inc/OpenAudioBench)
    - [VoxEval](https://huggingface.co/datasets/qqjz/VoxEval).
    - [seed-tts-eval](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit)
    - **storycloze**:
        - [sStoryCloze](https://drive.google.com/file/d/19ZnkM4vjApCZipd7xQ1ESlOi5oBVrlFL/view?usp=sharing)
        - [tStoryCloze](https://drive.google.com/file/d/17prYkldYb3w3Pyg3Pm77-VnE6nkD5jzG/view?usp=sharing)

Example cache directory structure for data:
```
<cache_dir>/
└── datas/
    ├── OpenAudioBench/
    ├── VoxEval/
    ├── seedtts_testset/
    ├── storycloze/
    │   ├── sSC/
    │   └── tSC/
    └── ... (other datasets)
```

## Model Preparation

- Models are typically downloaded and cached automatically into a `models` subdirectory within the specified cache directory (`--cache-dir`, defaults to `./cache`).
- Ensure the cache directory exists (e.g., create `./cache/models`).
- For models requiring manual download, place the model files within `<cache_dir>/models/`:
    - [WavLM-large fine-tuned](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)

Example cache directory structure for models:
```
<cache_dir>/
└── models/
    ├── WavLM-large-finetuned/
    │   └── ... (model files)
    └── ... (other models downloaded automatically or manually)
```

## Adding New Models

To add a new model to AudioCompass, follow these steps:

1. Create a new Python file in the `src/models` directory, e.g., `newmodel.py`
2. Implement a class that inherits from the `VoiceAssistant` base class:

```python
# src/models/newmodel.py
from .base import VoiceAssistant

class NewModelAssistant(VoiceAssistant):
    def __init__(self):
        # Initialize your model here
        pass
        
    def generate_a2t(self, audio, max_new_tokens=2048):
        pass
        
    def generate_t2t(self, text):
        pass
        
    def generate_at2t(self, audio, text, max_new_tokens=2048):
        pass

    # And other necessary methods for your model
```

3. Update the `src/models/__init__.py` file to import and register your new model:

```python
# In src/models/__init__.py
from .newmodel import NewModelAssistant

# Add to the model_cls_mapping dictionary
model_cls_mapping = {
    # existing models...
    'new_model': NewModelAssistant,
}
```

## Adding New Benchmarks

To add a new benchmark to AudioCompass, follow these steps:

1. Create a new directory in `src/benchmarks` for your benchmark, e.g., `src/benchmarks/newbenchmark/`
2. Implement a benchmark class that inherits from the `BaseBenchmark` class:

```python
# src/benchmarks/newbenchmark/newbenchmark.py
from ..base import BaseBenchmark

class NewBenchmark(BaseBenchmark):
    def __init__(self, subset_name, split):
        self.name = 'newbenchmark'
        self.subset_name = subset_name # if applicable
        self.split = split
        self.dataset = self.load_data()
    
    def load_data(self):
        # Load and preprocess your dataset here
        return dataset
    
    def generate(self, model):
        # Generate responses using the model
        return results
    
    def evaluate(self, data):
        # Implement evaluation metrics
        return evaluated_results
    
    def save_generated_results(self, results, output_dir, model_name):
        # Save generation results to the output directory
        pass
    
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.__class__.__name__)
        return self.evaluate(generated_results)
```

3. Update the `src/benchmarks/__init__.py` file to import and register your new benchmark:

```python
# In src/benchmarks/__init__.py
from .newbenchmark.newbenchmark import NewBenchmark

benchmark_mapping = {
    # existing benchmarks...
    'newbenchmark': NewBenchmark,
}
```
