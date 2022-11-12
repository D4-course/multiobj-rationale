""" ## !/usr/bin/env python """

from __future__ import print_function, division

import os
import pickle
import warnings

from chemprop.train import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data_from_smiles # , get_data
from chemprop.utils import load_args, load_checkpoint, load_scalers

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
# import rdkit.Chem.QED as QED
from rdkit.Chem import QED
# import scripts.sascorer as sascorer
from scripts import sascorer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


rdBase.DisableLog('rdApp.error')


class Gsk3Model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as fil:
            self.clf = pickle.load(fil)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        # for i,smiles in enumerate(smiles_list):
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fpind = Gsk3Model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fpind)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class Jnk3Model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as fil:
            self.clf = pickle.load(fil)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        # for i,smiles in enumerate(smiles_list):
        for smiles in smiles_list:  
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fpind = Jnk3Model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fpind)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class QedFunc():
    """ assigns scores to the molecule corresponding to the smile string """
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                scores.append(QED.qed(mol))
        return np.float32(scores)


class SaFunc():
    """ assigns scores to the molecule corresponding to the smile string """
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)

class ChempropModel():
    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    self.scaler, self.features_scaler = load_scalers(fname)
                    self.train_args = load_args(fname)
                    model = load_checkpoint(fname, cuda=True)
                    self.checkpoints.append(model)

    def __call__(self, smiles, batch_size=500):
        test_data = get_data_from_smiles(smiles=smiles,
                        skip_invalid_smiles=False, args=self.train_args)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)

        sum_preds = np.zeros((len(test_data), 1))
        for model in self.checkpoints:
            model_preds = predict(
                model=model,
                data=test_data,
                batch_size=batch_size,
                scaler=self.scaler
            )
            sum_preds += np.array(model_preds)

        # Ensemble predictions
        avg_preds = sum_preds / len(self.checkpoints)
        avg_preds = avg_preds.squeeze(-1).tolist()

        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]

        return np.array(full_preds, dtype=np.float32)


def get_scoring_function(prop_name):
    """Function that initializes and returns
     a scoring function by name"""
    if prop_name == 'jnk3':
        return Jnk3Model()
    if prop_name == 'gsk3':
        return Gsk3Model()
    if prop_name == 'qed':
        return QedFunc()
    if prop_name == 'sa':
        return SaFunc()
    # else:
    return ChempropModel(prop_name)

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
