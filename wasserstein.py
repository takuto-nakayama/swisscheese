from classes import Distance
import argparse
import warnings

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	warnings.filterwarnings('ignore', message='dgm1 has points with non-finite death times')

	parser.add_argument('dir_name')
	parser.add_argument('save_name')
	parser.add_argument('--range_samples',nargs='*',default=None)

	args = parser.parse_args()
	dir_name = args.dir_name
	save_name = args.save_name

	distance = Distance(
		dir_name=dir_name,
		save_name=save_name,
		)
	distance.get_wasserstein()
	distance.clustering()
	distance.msd_2d()