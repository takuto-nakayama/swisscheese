from classes import Distance
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dir_name')
	parser.add_argument('save_name')
	parser.add_argument('--p', type=int, default=2)
	parser.add_argument('--eps', type=float, default=0.01)
	parser.add_argument('--z', type=float, default=1.96)
	parser.add_argument('--n_direction', type=int, default=100)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--max_directions', type=int, default=10000)

	args = parser.parse_args()
	dir_name = args.dir_name
	save_name = args.save_name
	p = args.p
	eps = args.esp
	z = args.z
	n_directions = args.n_directions
	seed = args.seed
	max_directions = args.max_directions

	distance = Distance(
		dir_name=dir_name,
		save_name=save_name,
		)
	distance.get_sliced_wasserstein(
		p=p,
		eps=eps,
		z=z,
		n_directions=n_directions,
		seed=seed,
		max_directions=max_directions
		)
	distance.clustering()
	distance.msd_2d()