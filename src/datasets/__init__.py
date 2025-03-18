from .voicebench import VoiceBench

dataset_mapping = {
    'voicebench': VoiceBench,
}

def load_dataset(dataset_name, subset_name, split):
    """
    Load a dataset based on the provided dataset name, subset name, and split.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        subset_name (str): The name of the subset within the dataset.
        split (str): The split of the dataset to load (e.g., 'train', 'test').
    """
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    dataset_class = dataset_mapping[dataset_name]
    dataset_instance = dataset_class(subset_name, split)
    return dataset_instance.load_data()

def list_datasets():
    """
    List all available datasets.
    
    Returns:
        list: A list of available dataset names.
    """
    return list(dataset_mapping.keys())
