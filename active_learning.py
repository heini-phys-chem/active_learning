import sys
import subprocess
from ase import io
from ase.atoms import Atoms
from ase.calculators.ml import ML_calculator

from time import time
import numpy as np
from random import shuffle

import scipy
import scipy.stats

import numpy as np
from numpy.linalg import norm, inv

#import cPickle
import _pickle as cPickle

import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.math import qrlq_solve

from qml.representations import generate_fchl_acsf

from qml.kernels import get_local_kernels_gaussian
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_local_kernel
from qml.kernels import get_gdml_kernel
from qml.kernels import get_symmetric_gdml_kernel
from qml.kernels import get_local_gradient_kernel
from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel

sigma  = 2.5
num_it = 500

def read_file(f):
	lines = open(f, 'r').readlines()

	mols = np.array([])
	energies = np.array([])

	for line in lines:
		tokens   = line.split()
		mols     = np.append(mols, tokens[0])
		energies = np.append(energies, float(tokens[1]))

	return mols, energies

def read():
	alphas = np.load('alphas_active_learning.npy',allow_pickle=True)
	X      = np.load('X_active_learning.npy',allow_pickle=True)
	Q      = np.load('Q_active_learning.npy',allow_pickle=True)

	return alphas, X, Q

def predict(mols, alphas, X, Q):
	print("\n -> Start predictions")
	start = time()
	energies = np.array([])

	for mol in mols:
		alphas, X, Q = read()
		atoms = io.read("xyz/" + mol + ".xyz")
		atoms.set_calculator(ML_calculator(atoms, sigma, alphas, X, Q))

		energies = np.append(energies, atoms.get_potential_energy())

	end = time()
	total_runtime = end - start
	print("\n -> Prediction time: {:.3f}".format(total_runtime))

	return energies

def get_properties(filename):
    """ Returns a dictionary with energy and forces for each xyz-file.
    """
    # define dictionairies and constants
    properties = dict()
    path = "log/"
    # to convert Hartree/Bohr to (kcal/mol)/Angstrom
    conv = 27.2114/0.529

    # open file with names and energies
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    # loop through compounds
    for line in lines:
        forces		= np.array([])
        xyz				= np.array([])
        tokens		= line.split()

        name			= tokens[0]
        energy		= float(tokens[1])*27.2114
        #print(name, energy)

        # get xyz coordinates
        f_xyz			= open("xyz/" + name + ".xyz")
        ls			= f_xyz.readlines()
        f_xyz.close()

        for i,l in enumerate(ls):
          if i == 0:
            nAtoms = int(l)
          #if i == 1: continue
          #tokens	= l.split()
          #xyz			= np.append(xyz, [[tokens[1], tokens[2], tokens[3]]])
        #xyz				= xyz.reshape(nAtoms,3)

        # open orca output file to get the forces
        f_log			= open(path + name + ".log", "r")
        lines			= f_log.readlines()
        f_log.close()

        # find line with the final forces
        #index			= lines.index('The cartesian gradient:\n')
 #       index			= lines.index('The final MP2 gradient\n')
        index			= lines.index('CARTESIAN GRADIENT\n')

        # store forces in a 14x3 np.array
#        for line in lines[index+1:index+nAtoms+1]:
        for line in lines[index+3:index+nAtoms+3]:
          tokens	= line.split()
#          forces	= np.append(forces, [[float(tokens[1])*conv, float(tokens[2])*conv, float(tokens[3])*conv]])
          forces	= np.append(forces, [[float(tokens[3])*conv, float(tokens[4])*conv, float(tokens[5])*conv]])
        forces		= forces.reshape(nAtoms,3)

        # store name, energy and forces in a dictionairy
        xyz = []
        properties[name] = [energy, forces, xyz]

    return properties

def train():
#	print(" -> Start training")
#	start = time()
#	subprocess.Popen(("python3","model_training.py","train"))
#	end = time()
#
#	total_runtime = end - start
#
#	print(" -> Training time: {:.3f}".format(total_runtime))
    #data = get_properties("energies.txt")
    data = get_properties("train")
    mols = []
    mols_pred = []

    SIGMA = 2.5 #float(sys.argv[1])

    for name in sorted(data.keys()):
        mol = qml.Compound()
        mol.read_xyz("xyz/" + name + ".xyz")

        # Associate a property (heat of formation) with the object
        mol.properties = data[name][0]
        mols.append(mol)


    shuffle(mols)

    #mols_train = mols[:400]
    #mols_test = mols[400:]

    # REPRESENTATIONS
    print("\n -> calculate representations")
    start = time()
    x = []
    disp_x = []
    f = []
    e = []
    q = []

    for mol in mols:
      (x1, dx1) = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=True, pad=23, elements=[1,6,7,8,16,17])

      e.append(mol.properties)
      f.append( data[(mol.name)[4:-4]][1])
      x.append(x1)
      disp_x.append(dx1)
      q.append(mol.nuclear_charges)

    X_train = np.array(x)
    F_train = np.array(f)
    F_train *= -1
    E_train = np.array(e)
    dX_train = np.array(disp_x)
    Q_train = q

    E_mean = np.mean(E_train)

    E_train -= E_mean

    F_train = np.concatenate(F_train)

    end = time()

    print(end-start)
    print("")
    print(" -> calculating Kernels")

    start = time()
    Kte = get_atomic_local_kernel(X_train,  X_train, Q_train,  Q_train,  SIGMA)
    #Kte_test = get_atomic_local_kernel(X_train,  X_test, Q_train,  Q_test,  SIGMA)

    Kt = get_atomic_local_gradient_kernel(X_train,  X_train, dX_train,  Q_train,  Q_train, SIGMA)
    #Kt_test = get_atomic_local_gradient_kernel(X_train,  X_test, dX_test,  Q_train,  Q_test, SIGMA)

    C = np.concatenate((Kte, Kt))

    Y = np.concatenate((E_train, F_train.flatten()))
    end = time()
    print(end-start)
    print("")

    print("Alphas operator ...")
    start = time()
    alpha = svd_solve(C, Y, rcond=1e-12)
    end = time()
    print(end-start)
    print("")

    print("save X")
    np.save('X_active_learning.npy', X_train)
#    with open("X_mp2.cpickle", 'wb') as f:
#      cPickle.dump(X_train, f, protocol=2)

    print("save alphas")
    np.save('alphas_active_learning.npy', alpha)
#    with open("alphas_mp2.cpickle", 'wb') as f:
#      cPickle.dump(alpha, f, protocol=2)

    print("save Q")
    np.save('Q_active_learning.npy', Q_train)
#    with open("Q_mp2.cpickle", 'wb') as f:
#      cPickle.dump(Q_train, f, protocol=2)

    eYt = np.dot(Kte, alpha)
    fYt = np.dot(Kt, alpha)
    #eYt_test = np.dot(Kte_test, alpha)
    #fYt_test = np.dot(Kt_test, alpha)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E_train, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E_train - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F_train.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F_train.flatten() - fYt.flatten())), slope, intercept, r_value ))



def add_candidate(mol, energy):
#	for i, mol in enumerate(mols):
	f = open("train", 'a')
	f.write(mol + " " + str(energy) + "\n")
	f.close()

	print(" -> added: {}".format(mol))

def remove_from_mols(mol, index):
#	for i, mol in enumerate(mols):
	np.delete(mol, index)

	return mols


if __name__ == "__main__":

	# read in mols to predict
	filename = sys.argv[1]
	mols, mp2_energies = read_file(filename)

	# initial training
	train()

	# loop over iteration steps
	for i in range(num_it):
		alphas, X, Q = read()
		energies = predict(mols, alphas, X, Q)

		MAEs = np.array([])

		# calculate MAEs
		for j, energy in enumerate(energies):
			MAEs = np.append(MAEs, np.abs(energy - mp2_energies[j]))

		# get mean of MAEs
		mean = np.mean(MAEs)

		# exit the loop of chemical accuracy (1kcal/mol) is reached
		if mean < .6:
			print("Chemical Accuracy reached ({:.4f} kcal/mol) with {:d} iterations".format(mean, i))
			exit()

		# get worst prediction
		index = np.argmin(MAEs)
		mol_to_add = mols[index]
		# add worst predicted candidate to teh training
		add_candidate(mol_to_add, mp2_energies[index])
		# remove worst candidate from the rpediction list (mols)
		mols = remove_from_mols(mols, index)
		# retrain
		train()
