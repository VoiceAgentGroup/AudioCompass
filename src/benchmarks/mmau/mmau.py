from loguru import logger
from tqdm import tqdm
import json
import os
from evaluate import _evaluate
from base import BaseBenchmark


class MMAU(BaseBenchmark):
    def __init__(self, subset_name, split):
        self.name = 'mmau'
        self.subset_name = subset_name
        self.split = split
        self.dataset = self.load_data()