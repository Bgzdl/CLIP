import argparse

parser = argparse.ArgumentParser(description="Model training hyperparameters")

parser.add_argument('epoches', type=int, help='Number of epoches')
parser.add_argument('batch_size', type=int, help='Batch size')
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument('temperature', type=float, help='Temperature')
parser.add_argument('decayRate', type=float, help='Weight decay')
