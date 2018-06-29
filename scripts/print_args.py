import argparse
import torch

"""
Tiny utility to print the command-line args used for a checkpoint
"""

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')


def main(args):
	checkpoint = torch.load(args.checkpoint, map_location='cpu')
	for k, v in checkpoint['args'].items():
		print(k, v)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
