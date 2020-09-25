#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rnn import CharRNN

module_dir = os.path.dirname(__file__)
dataset_path = os.path.join(module_dir, 'dataset.txt')

n_hidden=512
n_layers=3
train_on_gpu = torch.cuda.is_available()
f = open(dataset_path, 'r', encoding='utf-8')
sonnets_dataset = f.read().split('<eos>')
del sonnets_dataset[-1]
dataset_string = ''
for line in sonnets_dataset:
    dataset_string = dataset_string + line
chars = tuple(set(dataset_string))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in dataset_string])

model_path = os.path.join(module_dir, 'model_try.pt')
neuralnetwork = CharRNN(chars, n_hidden, n_layers)

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sonnetai.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
