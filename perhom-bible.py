from classes import Embedding, PersistenceDiagram
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('model_name')
	parser.add_argument('text_name')
	parser.add_argument('save_name')
	parser.add_argument('lang', default=None)
	parser.add_argument('batch', default=100)

	args = parser.parse_args()
	model_name = args.model_name
	text_name = args.text_name
	save_name = args.save_name
	lang = args.lang
	batch = args.batch

	embedding = Embedding(model_name=model_name, lang=lang)
	embedding.embed_dynamic(file_path=text_name, batch=batch)

	pedg = PersistenceDiagram()
	pedg.pers_homology(file_path=save_name)