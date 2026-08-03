from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('model_name')
	parser.add_argument('config')
	parser.add_argument('save_name')
	parser.add_argument('--batch', default=100)
	parser.add_argument('--seed', default=42)

	args = parser.parse_args()
	model_name = args.model_name
	config = args.config
	save_name = args.save_name
	batch = args.batch
	seed = args.seed


	embedding = Embedding(model_name=model_name)
	embedding.embed_dynamic_wiki(config=config, batch=batch, seed=seed)

	pedg = PersistenceDiagram(embedding.embeddings)
	pedg.pers_homology(file_name=f'{save_name}')