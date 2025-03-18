from datasets import load_dataset, Audio

class VoiceBench:
    def __init__(self, subset_name, split):
        self.subset_name = subset_name
        self.split = split

    def load_data(self):
        dataset = load_dataset('hlt-lab/voicebench', self.subset_name, split=self.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        return dataset