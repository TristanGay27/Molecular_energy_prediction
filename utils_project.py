import pandas as pd 
import os
import numpy as np
from pathlib import Path
from ase.io import read

def generate_csv(ids, energies, method):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "results", f"{method}.csv")
    df = pd.DataFrame({'id': ids, 'energy': energies})
    df.to_csv(output_path, header=True, index=False)

def extract_features_from_xyz_inv(file_path):
    atoms = read(file_path)

    positions = atoms.get_positions()
    masses = atoms.get_masses()
    atomic_numbers = atoms.get_atomic_numbers()
    num_atoms = len(atoms)

    # Total mass
    total_mass = np.sum(masses)

    # Distance matrix
    distances = atoms.get_all_distances(mic=False)
    tril_indices = np.tril_indices(num_atoms, k=-1)
    interatomic_distances = distances[tril_indices]

    mean_distance = np.mean(interatomic_distances)
    min_distance = np.min(interatomic_distances)
    max_distance = np.max(interatomic_distances)
    bond_length_std = np.std(interatomic_distances)

    # Radius of gyration
    center_of_mass = np.average(positions, axis=0, weights=masses)
    rel_positions = positions - center_of_mass
    squared_distances = np.sum(rel_positions**2, axis=1)
    radius_of_gyration = np.sqrt(np.sum(squared_distances * masses) / total_mass)

    # Inertia tensor (rotation invariant)
    inertia_tensor = np.zeros((3, 3))
    for i in range(num_atoms):
        r = rel_positions[i]
        m = masses[i]
        inertia_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    eigvals = np.linalg.eigvalsh(inertia_tensor)
    eigvals.sort()

    features = {
        'num_atoms': num_atoms,
        'total_mass': total_mass,
        'mean_distance': mean_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'bond_length_std': bond_length_std,
        'radius_of_gyration': radius_of_gyration,
        'inertia_eig_0': eigvals[0],
        'inertia_eig_1': eigvals[1],
        'inertia_eig_2': eigvals[2],
    }

     # === Matrice de Coulomb ===
    Z = atoms.get_atomic_numbers()
    N = len(Z)
    coulomb = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                coulomb[i, j] = 0.5 * Z[i] ** 2.4
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
                coulomb[i, j] = Z[i] * Z[j] / dist if dist != 0 else 0

    # Coulomb features: stats
    coulomb_no_diag = coulomb[~np.eye(N, dtype=bool)]
    coulomb_stats = {
        'coulomb_mean': np.mean(coulomb_no_diag),
        'coulomb_std': np.std(coulomb_no_diag),
        'coulomb_max': np.max(coulomb_no_diag),
        'coulomb_min': np.min(coulomb_no_diag),
    }

    # Coulomb spectrum (valeurs propres triées)
    spectrum = np.linalg.eigvalsh(coulomb)
    spectrum = np.sort(spectrum)[::-1]  # tri décroissant
    top_10_spectrum = spectrum[:10]  # garder les 10 premiers

    spectrum_features = {f'coul_spec_{i}': val for i, val in enumerate(top_10_spectrum)}

    features.update(coulomb_stats)
    features.update(spectrum_features)

    return features


def extract_features_from_xyz(file_path):
    atoms = read(file_path)

    num_atoms = len(atoms)
    atom_types = atoms.get_chemical_symbols()
    unique_types, counts = np.unique(atom_types, return_counts=True)
    type_counts = {type_: count for type_, count in zip(unique_types, counts)}

    positions = atoms.get_positions()
    masses = atoms.get_masses()
    center_of_mass = np.average(positions, axis=0, weights=masses)

    distances_from_com = positions - center_of_mass
    squared_distances = np.sum(distances_from_com**2, axis=1)
    radius_of_gyration = np.sqrt(np.sum(squared_distances * masses) / np.sum(masses))

    distances = atoms.get_all_distances()
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    features = {
        'num_atoms': num_atoms,
        'radius_of_gyration': radius_of_gyration,
        'mean_distance': mean_distance,
        'max_distance': max_distance,
        'min_distance': min_distance,
        'center_of_mass_x': center_of_mass[0],
        'center_of_mass_y': center_of_mass[1],
        'center_of_mass_z': center_of_mass[2]
        # Add more features as needed
    }

    return features


def extract_positions_charges_energies_from_xyz(xyz_file,max_atoms):
    # Read the XYZ file using ASE
    atoms = read(xyz_file)

    # Extract positions
    positions = atoms.get_positions().tolist()
    padding = [[0.0, 0.0, 0.0] for _ in range(max_atoms - len(positions))]
    positions.extend(padding)

    # Extract charges if they exist; otherwise, use a default value of 0.0
    charges = atoms.get_atomic_numbers().tolist()
    charges.extend([0 for _ in range(max_atoms - len(charges))])


    return {
        'positions': np.array(positions),
        'charges': np.array(charges),
    }

def create_dataframe_from_xyz_files(path, csv_path=None,inv_only=False):
    
    if csv_path is not None:
        energy_data = pd.read_csv(csv_path)

    data_dir = Path(path)
    xyz_files = sorted(list(data_dir.glob("*.xyz")))
    data = []

    max_atoms = 0
    for xyz_file in xyz_files:
        atoms = read(xyz_file)
        num_atoms = len(atoms)
        if num_atoms > max_atoms:
            max_atoms = num_atoms

    for xyz_file in xyz_files:
        molecule_id = xyz_file.stem
        numeric_id = int(molecule_id.split('_')[-1])
        features=[]
        if inv_only:
            features = extract_features_from_xyz_inv(xyz_file)
        else:
            features = extract_features_from_xyz(xyz_file)
        features['id'] = numeric_id

        # Extract positions, charges, and energy from the XYZ file
        additional_features = extract_positions_charges_energies_from_xyz(xyz_file,max_atoms)

        if csv_path is not None:
            energy = energy_data.loc[energy_data['id'] == numeric_id, 'energy'].values
            if len(energy) > 0:
                additional_features['energy'] = energy[0]

        # Combine the features
        combined_features = {**features, **additional_features}
        data.append(combined_features)

    df = pd.DataFrame(data).fillna(0)
    return df





def create_X_y_from_dataframe(df, feature_columns=None, test=False):
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != 'energy']

    X = np.asarray(df[feature_columns])
    y = None
    if not test and 'energy' in df.columns:
        y = np.asarray(df['energy'])

    return X, y


