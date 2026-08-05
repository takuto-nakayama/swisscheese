from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('model_name')
	parser.add_argument('config')
	parser.add_argument('save_name')
	parser.add_argument('--batch', default=100)
	parser.add_argument('--num_samples', default=5000)
	parser.add_argument('--seed_range', default=10)

	args = parser.parse_args()
	model_name = args.model_name
	config = args.config
	save_name = args.save_name
	batch = args.batch
	num_samples = args.num_samples
	seed_range = args.seed_range


	embedding = Embedding(model_name=model_name)
	embedding.embed_dynamic_wiki(config=config, batch=batch, num_samples=num_samples, seed=42)

	for seed in range(seed_range):
		pedg = PersistenceDiagram(embeddings=embedding.embeddings, seed=seed, num_samples=num_samples)
		pedg.pers_homology(file_name=f'{save_name}-{seed}')