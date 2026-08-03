from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('lang')
	parser.add_argument('id')
	parser.add_argument('seed')
	parser.add_argument('save_name')

	args = parser.parse_args()
	lang = args.lang
	id = args.id
	seed = args.seed
	save_name = args.save_name

	embedding = Embedding(lang=id)
	embedding.embed_fasttext_model(seed=seed)

	pedg = PersistenceDiagram(embeddings=embedding.embeddings)
	pedg.pers_homology(file_path=save_name)