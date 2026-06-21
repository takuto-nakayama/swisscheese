from classes import PersistenceDiagram
from dotenv import load_dotenv
import argparse, os

if __name__=='__main__':
	#  load the environment variables
	load_dotenv()
	data_dir = os.getenv('DATA_DIR')
	emb_dir = os.getenv('EMB_DIR')
	pd_dir = os.getenv('PD_DIR')
	parser = argparse.ArgumentParser()


	#  parse the arguments
	parser.add_argument('path_embedding',
					    type=str)
	parser.add_argument('dataset_name',
					    type=str)
	parser.add_argument('-cycle',
					    type=int,
						default=1)
	parser.add_argument('-thresh',
					    type=float,
						default=1.0)
	parser.add_argument('-n_perm',
					    type=int,
						default=1000)
	parser.add_argument('-save_pd',
					    action='store_true')
	parser.add_argument('-save_path_pd',
					    type=str,
						default='perhom')
	parser.add_argument('-save_ws',
						action='store_true')

	args = parser.parse_args()
	path_embedding =	args.path_embedding
	dataset_name =		args.dataset_name
	cycle =				args.cycle
	thresh =			args.thresh
	n_perm =			args.n_perm
	save_pd =			args.save_pd
	save_path_pd =		args.save_path_pd
	save_ws =			args.save_ws


	#  main process
	## gain the persistent homology
	for i in range(1,cycle+1):
		pd = PersistenceDiagram(file_path=f'{emb_dir}/{path_embedding}',
								dataset=dataset_name)
		
		pd.pers_hom(thresh=thresh,
					n_perm=n_perm,
					save=save_pd,
					file_path=f'{pd_dir}/{save_path_pd}-{i}')
		
