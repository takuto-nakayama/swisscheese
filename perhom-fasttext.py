from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('id')
	parser.add_argument('save_name')
	parser.add_argument('--num_samples', type=int, default=5000)
	parser.add_argument('--seed', type=int, default=42)

	args = parser.parse_args()
	id = args.id
	save_name = args.save_name
	num_samples = args.num_samples
	seed = args.seed

	embedding = Embedding()
	embedding.embed_fasttext_model(id=id, num_samples=num_samples, seed=seed)

	pedg = PersistenceDiagram(embeddings=embedding.embeddings)
	pedg.pers_homology(file_path=save_name)