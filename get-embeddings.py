from classes import Embedding, PersistenceDiagram
from dotenv import load_dotenv
import argparse, os

if __name__=='__main__':
	#  load the environment variables
	load_dotenv()
	data_dir = os.getenv('DATA_DIR')
	result_dir = os.getenv('RESULT_DIR')
	parser = argparse.ArgumentParser()

	#  parse the arguments
	parser.add_argument('path_data')
	parser.add_argument('-model_name',
					 	type=str,
						default='bert-base-multilingual-uncased')
	parser.add_argument('-batch',
					    type=int,
						default=500)
	parser.add_argument('-lang',
					    type=str,
						default=None)
	parser.add_argument('-save_emb',
					    action='store_true')
	parser.add_argument('-save_path_emb',
					 	type=str,
						default='embedding')
	parser.add_argument('-dataset_name',
					    type=str,
						default='dataset')
	

	args = parser.parse_args()
	model_name =	args.model_name
	batch =			args.batch
	lang =			args.lang
	path_data =		args.path_data
	save_emb =		args.save_emb
	save_path_emb =	args.save_path_emb
	dataset_name =	args.dataset_name


	#  main process
	## gain the embeddings from the input data
	embedding = Embedding(model_name=model_name,
					      lang=lang)
	embedding.embed(file_path=f'{data_dir}/{path_data}',
				    batch=batch)
	if save_emb:
		embedding.save(file_path=f'{result_dir}/{save_path_emb}',
					   dataset=dataset_name)
