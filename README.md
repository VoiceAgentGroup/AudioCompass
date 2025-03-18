# AudioCompass

AudioCompass is a comprehensive evaluation framework for audio and multimodal language models. This platform allows researchers and developers to benchmark various voice assistant models using standardized datasets and metrics.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
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
git clone https://github.com/wtalioy/AudioCompass.git
cd AudioCompass

# Set up the environment
conda create -n voicebench python=3.10
conda activate voicebench
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23 --no-deps
pip install -r requirements.txt
```

## Usage and Evaluation

To run a benchmark on a specific model:

```bash
python main.py --model <model_name> --benchmark <dataset_name> --subset <subset_name> --split <split_name> --output-dir <output_directory>
```

For example:

```bash
# local is for models running on localhost
python main.py --model local --benchmark voicebench --subset alpacaeval --split test --output-dir output
```

The evaluation process:
1. Loads the specified model and benchmark
2. Generates responses for the benchmark dataset using the model
3. Saves the generation results to the specified output directory (typically as JSONL files)
4. Evaluates the responses using appropriate metrics
5. Returns and prints the evaluation results to the console

To list available models and benchmarks:

```python
from src.models import list_models
print(list_models())

from src.benchmarks import list_benchmarks
print(list_benchmarks())
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
        
    def generate_audio(self, audio, max_new_tokens=2048):
        # Implement audio-only input processing
        # audio: dict with 'array' and 'sampling_rate' keys
        # Return text response
        pass
        
    def generate_text(self, text):
        # Implement text-only input processing
        # Return text response
        pass
        
    def generate_mixed(self, audio, text, max_new_tokens=2048):
        # Implement mixed input processing
        # audio: dict with 'array' and 'sampling_rate' keys
        # text: string input
        # Return text response
        pass
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
