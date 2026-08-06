from classes import Embedding, PersistenceDiagram
import numpy as np
import argparse, sys

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('save_name')
	parser.add_argument(
		'--num_embeddings',
		type=int,
		default=5000
		)
	parser.add_argument(
		'--dim',
		type=int,
		default=768
		)
	parser.add_argument(
		'--mode',
		type=str,
		default='rand'
		)
	parser.add_argument(
		'--seed',
		type=int,
		default=42)

	args = parser.parse_args()
	save_name = args.save_name
	dim = args.dim
	num_embeddings = args.num_embeddings
	mode = args.mode
	seed = args.seed


	rng = np.random.default_rng(seed)
	if mode == 'rand':
		print(f'mode is {mode}. shape={num_embeddings, dim}')
		embeddings = rng.random((num_embeddings, dim))
	elif mode == 'normal':
		print(f'mode is {mode}. shape={num_embeddings, dim}')
		embeddings = rng.normal(size=(num_embeddings, dim))
	else:
		sys.exit('mode must be "rand" or "normal"!')
		
	pedg = PersistenceDiagram(embeddings=embeddings)
	pedg.pers_homology(file_name=f'{save_name}')