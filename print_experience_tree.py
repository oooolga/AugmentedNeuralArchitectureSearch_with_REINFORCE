import argparse

from src.model.experience_replay import recursivePrintTree
from src.util.util import load_reinforce_model
from src.model.layers import LAYERS_TYPE, NUM_LAYERS_TYPE

def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--load-model', required=True, type=str, help='Path for load model')

	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse()

	replay_tree, _, _, _, _, _, _, _ = load_reinforce_model(args.load_model)

	recursivePrintTree(replay_tree)

	for layer_idx, layer_t in enumerate(LAYERS_TYPE):
		print('{}: {}'.format(layer_idx, layer_t))