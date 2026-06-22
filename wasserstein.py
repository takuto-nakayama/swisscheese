from classes import Distance
from dotenv import load_dotenv
import argparse, os
import warnings

if __name__=='__main__':
	#  load the environment variables
	load_dotenv()
	pd_dir = os.getenv('PD_DIR')
	ws_dir = os.getenv('WS_DIR')
	parser = argparse.ArgumentParser()
	warnings.filterwarnings('ignore', message='dgm1 has points with non-finite death times')


	#  parse the arguments
	parser.add_argument('pd_path',
					    type=str)
	parser.add_argument('save_path_pd',
					    type=str)

	args = parser.parse_args()
	pd_path			= args.pd_path
	save_path_pd	= args.save_path_pd


	#  main process
	## gain the wasserstein distance
	distance = Distance(pd_path=f'{pd_dir}/{pd_path}',
					    file_path=f'{ws_dir}/{save_path_pd}')
	distance.get_wasserstein()