from classes import Distance
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dir_name')
	parser.add_argument('save_name')

	args = parser.parse_args()
	dir_name = args.dir_name
	save_name = args.save_name

	distance = Distance(
		dir_name=dir_name,
		save_name=save_name,
		)
	distance.get_sliced_wasserstein()
	distance.clustering()
	distance.msd_2d()