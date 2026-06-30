"""
==============================================================================
SVD ENGINE — Phase 1: The Core Math Engine (Custom SVD from Scratch)
==============================================================================

WHAT THIS FILE DOES:
    Implements Singular Value Decomposition (SVD) from scratch using only
    NumPy. No np.linalg.svd allowed here — we build it ourselves.

THE MATH IN PLAIN ENGLISH:
    SVD breaks any matrix A into three pieces:  A = U · Σ · Vᵀ
    
    - U  = Left singular vectors  (column directions of A)
    - Σ  = Singular values        (how "important" each direction is)
    - Vᵀ = Right singular vectors (row directions of A)
    
    To find these, we use a technique called EIGENDECOMPOSITION:
    - AᵀA gives us V (right singular vectors) and σ² (squared singular values)
    - AAᵀ gives us U (left singular vectors) and σ² (same squared singular values)
    
    To find eigenvalues/eigenvectors, we use POWER ITERATION:
    - Multiply a random vector by the matrix over and over
    - It naturally aligns with the dominant eigenvector
    - Then use DEFLATION to strip that component and find the next one

BUILD ORDER:
    1. power_iteration()   — finds ONE dominant eigenvector
    2. deflation()         — removes that eigenvector's influence
    3. eigen_decompose()   — uses 1 & 2 in a loop to find k eigenvectors
    4. custom_svd()        — uses 3 to compute full SVD
==============================================================================
"""

# pyrefly: ignore [missing-import]
import numpy as np

def power_iteration(M, num_iterations=100):
    """Find the dominant eigenvalue and eigenvector using Power Iteration."""
    n = M.shape[0]
    b = np.random.rand(n)
    for _ in range(num_iterations):
        b_new = np.dot(M,b)
        b = b_new / np.linalg.norm(b_new)
    eigenvalue = np.dot(b.T,np.dot(M,b))
    return eigenvalue,b

def deflation(M, eigenvector, eigenvalue):
    """Remove the dominant eigencomponent from matrix M."""
    outer = np.outer(eigenvector, eigenvector)
    M_deflated = M - eigenvalue * outer
    return M_deflated

def eigen_decompose(M, k, num_iterations=100):
    """Find the top-k eigenvalues and eigenvectors of a symmetric matrix."""    
    eigenvalues = []
    eigenvectors = []
    M_current = M.copy()
    for _ in range(k):
        eigenvalue,eigenvector = power_iteration(M_current,num_iterations)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        M_current = deflation(M_current,eigenvector,eigenvalue)
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.column_stack(eigenvectors)
    return eigenvalues,eigenvectors

def custom_svd(A, k):
    """Compute rank-k SVD decomposition of matrix A from scratch."""
    AtA = np.dot(A.T,A)
    eigenvalues,V = eigen_decompose(AtA,k)
    eigenvalues = np.maximum(eigenvalues,0)
    sigma = np.sqrt(eigenvalues)
    m,n = A.shape
    U = np.zeros((m,k))
    for i in range(k):
        if sigma[i] > 1e-10:
            U[:,i] = np.dot(A,V[:,i])/sigma[i]
    Vt = V.T
    return U, sigma, Vt

def reconstruct_svd(U,sigma,Vt):
    """Reconstruct image from SVD components."""
    return U @ np.diag(sigma) @ Vt

