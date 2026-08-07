from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('id')
	parser.add_argument('save_name')
	parser.add_argument('--num_points', type=int, default=5000)
	parser.add_argument('--seed_range', type=int, default=10)

	args = parser.parse_args()
	id = args.id
	save_name = args.save_name
	num_points = args.num_points
	seed_range = args.seed_range

	embedding = Embedding()
	embedding.embed_fasttext_model(
		id=id,
		num_samples=num_samples,
		seed=42
		)

	for seed in range(seed_range):
		pedg = PersistenceDiagram(
			embeddings=embedding.embeddings,
			seed=seed,
			num_points=num_points,
			)
		pedg.pers_homology(file_path=save_name)