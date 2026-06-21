from classes import Distance
from dotenv import load_dotenv
import argparse, os

if __name__=='__main__':
	#  load the environment variables
	load_dotenv()
	pd_dir = os.getenv('PD_DIR')
	ws_dir = os.getenv('WS_DIR')
	parser = argparse.ArgumentParser()


	#  parse the arguments
	parser.add_argument('save_path_pd',
					    type=str,)

	args = parser.parse_args()
	save_path_pd =	args.save_path_pd


	#  main process
	## gain the wasserstein distance
	distance = Distance(pds_path=pd_dir,
					    file_path=f'{ws_dir}/{save_path_pd}')
