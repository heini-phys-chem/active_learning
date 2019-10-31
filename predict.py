''' ML calculator'''
import sys
import numpy as np
from ase import io
from ase.atoms import Atoms
from ase.calculators.ml import ML_calculator

sigma  = 2.5
alphas = np.load('alphas.npy',allow_pickle=True)
X      = np.load('X.npy',allow_pickle=True)
Q      = np.load('Q.npy',allow_pickle=True)
alphas_active_learning = np.load('alphas_active_learning.npy',allow_pickle=True)
X_active_learning      = np.load('X_active_learning.npy',allow_pickle=True)
Q_active_learning      = np.load('Q_active_learning.npy',allow_pickle=True)

def read_file(f):
	lines = open(f, 'r').readlines()

	names    = np.array([])
	energies = np.array([])

	for line in lines:
		tokens = line.split()

		names    = np.append(names, tokens[0])
		energies = np.append(energies, float(tokens[1]))

	return names, energies

if __name__ == "__main__":

	filename = sys.argv[1]
	mols, energies = read_file(filename)

	MAEs = np.array([])

	for i, mol in enumerate(mols):
		atoms = io.read("xyz/" + mol + ".xyz")
		atoms.set_calculator(ML_calculator(atoms, sigma, alphas, X, Q))
		energy = atoms.get_potential_energy()

		MAE = np.abs(energy - energies[i])
		MAEs = np.append(MAEs, MAE)

	mean = np.mean(MAEs)

	print("mean MAE is: {}".format(mean))

	for i, mol in enumerate(mols):
		atoms = io.read("xyz/" + mol + ".xyz")
		atoms.set_calculator(ML_calculator(atoms, sigma, alphas_active_learning, X_active_learning, Q_active_learning))
		energy = atoms.get_potential_energy()

		MAE = np.abs(energy - energies[i])
		MAEs = np.append(MAEs, MAE)

	mean = np.mean(MAEs)

	print("mean MAE (active learning) is: {}".format(mean))
