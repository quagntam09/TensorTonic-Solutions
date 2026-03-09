import numpy as np

def matrix_transpose(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype = np.float64)
    N, M = A.shape
    AT = np.zeros((M, N), dtype=A.dtype)
    for i in range(N):
        for j in range(M):
            AT[j, i] = A[i, j]
    
    return AT