from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('model_name')
	parser.add_argument('text_name')
	parser.add_argument('save_name')
	parser.add_argument('--batch', default=100)
	parser.add_argument('--seed_range', default=10)

	args = parser.parse_args()
	model_name = args.model_name
	text_name = args.text_name
	save_name = args.save_name
	batch = args.batch
	seed_range = args.seed_range


	embedding = Embedding(model_name=model_name)
	embedding.embed_dynamic(file_path=text_name, batch=batch)

	for seed in range(seed_range):
		pedg = PersistenceDiagram(seed=seed)
		pedg.pers_homology(file_path=save_name)