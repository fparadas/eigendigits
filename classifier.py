import numpy as np
from typing import NamedTuple
from tqdm import tqdm


class Model:
    label_list: np.ndarray
    weights: np.ndarray

    def fit(self, X: np.ndarray, Y: np.ndarray, k: int = 5):
        label_list = np.unique(Y)

        # Reshape into data matrices and group by labels
        X_reshaped = X.reshape(-1, X.shape[1] * X.shape[2]).T

        data_matrices = [X_reshaped[:, Y == label] for label in label_list]

        U = [0] * len(label_list)
        S = [0] * len(label_list)
        Vh = [0] * len(label_list)

        # Perform SVD to get the Eigenstuff vector
        for index in tqdm(range(len(label_list))):
            u, s, vh = np.linalg.svd(data_matrices[index], full_matrices=False)
            U[index] = u
            S[index] = s
            Vh[index] = vh

        U = np.array(U)
        print(U.shape)
        # U is now a tensor, where the first dim is the labels,
        # the second and third are the matrix u from the svd algorithm,
        # which encode the dataset features or the "Eigenstuffs"

        I = np.identity(X.shape[1] * X.shape[2])

        # Calculate the projection matrices for every label up to the kth eigeinvalues vector
        projection_matrices = np.einsum("ijk,ilk->ijl", U[:, :, :k], U[:, :, :k])

        self.label_list = label_list
        self.weights = I - projection_matrices
        self.U = U
        self.S = S
        self.Vh = Vh

    def predict(self, Z, depth=5):
        Z.reshape(-1, Z.shape[1] * Z.shape[2]).T
        projection_residuals = np.tensordot(self.weights, Z, axes=([1], [0]))

        return self.label_list[
            np.argmin(np.linalg.norm(projection_residuals, ord=2, axis=1), axis=0)
        ]
