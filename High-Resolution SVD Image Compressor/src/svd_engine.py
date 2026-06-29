"""
==============================================================================
SVD ENGINE — Phase 1: The Core Math Engine (Custom SVD from Scratch)
==============================================================================

This is the HEART of the entire project. Everything else depends on this.

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

NOTEBOOK TEST (after completing this file):
    In Notebook.ipynb, Section 1:
    - Test power_iteration on a 4×4 matrix, watch eigenvalue converge
    - Test deflation, verify 2nd eigenvalue < 1st
    - Test custom_svd, compare U, Σ, V with np.linalg.svd output
    - Test on a 32×32 random matrix (actual block size we'll use later)
==============================================================================
"""

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


# ==============================================================================
# STEP 1.3 (HELPER): EIGEN DECOMPOSITION
# ==============================================================================
#
# WHAT IT DOES:
#     Finds the top-k eigenvalues and eigenvectors of a symmetric matrix
#     by calling power_iteration and deflation in a loop.
#
# THE ALGORITHM:
#     1. Run power_iteration → get eigenvalue_1, eigenvector_1
#     2. Run deflation → get M without eigenvalue_1's influence
#     3. Run power_iteration on deflated M → get eigenvalue_2, eigenvector_2
#     4. Repeat until we have k eigenvalues and eigenvectors
#
# PARAMETERS:
#     M              : numpy 2D array — a square symmetric matrix
#     k              : int — how many eigenvalues/eigenvectors to find
#     num_iterations : int — iterations for each power_iteration call
#
# RETURNS:
#     eigenvalues  : numpy 1D array of shape (k,)  — the top-k eigenvalues (descending)
#     eigenvectors : numpy 2D array of shape (n, k) — each column is an eigenvector
#
# HINTS:
#     - Start with empty lists for eigenvalues and eigenvectors
#     - In each loop iteration: find eigenpair → store it → deflate the matrix
#     - Use np.array() to convert lists to arrays at the end
#     - Stack eigenvectors as columns: np.column_stack(eigenvector_list)
#
def eigen_decompose(M, k, num_iterations=100):
    """Find the top-k eigenvalues and eigenvectors of a symmetric matrix."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Create empty lists to collect eigenvalues and eigenvectors
    # HINT: eigenvalues = []
    # HINT: eigenvectors = []
    
    # Step 2: Make a copy of M so we don't modify the original
    # HINT: M_current = M.copy()
    
    # Step 3: Loop k times:
    #     a) Call power_iteration(M_current, num_iterations)
    #        → get (eigenvalue, eigenvector)
    #     b) Append eigenvalue to eigenvalues list
    #     c) Append eigenvector to eigenvectors list
    #     d) Call deflation(M_current, eigenvector, eigenvalue)
    #        → update M_current with the deflated matrix
    
    # Step 4: Convert to numpy arrays and return
    # HINT: eigenvalues = np.array(eigenvalues)
    # HINT: eigenvectors = np.column_stack(eigenvectors)
    # HINT: return eigenvalues, eigenvectors
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 1.4: THE SVD WRAPPER — custom_svd(A, k)
# ==============================================================================
#
# WHAT IT DOES:
#     Computes the rank-k SVD of ANY matrix A (not necessarily square).
#     Returns U, sigma, Vt such that A ≈ U · diag(sigma) · Vt
#
# THE ALGORITHM:
#     1. Compute AᵀA  (this is a square symmetric matrix)
#     2. Run eigen_decompose(AᵀA, k) to get:
#        - eigenvalues (these are σ² — the squared singular values)
#        - V (the right singular vectors — columns of V)
#     3. Compute singular values: σ = √eigenvalues
#     4. Compute U from the formula:  u_i = (A · v_i) / σ_i
#        (each column of U = A times the corresponding column of V, divided by σ)
#     5. Return U, sigma, Vᵀ
#
# WHY NOT JUST USE AAᵀ FOR U?
#     We COULD run eigen_decompose(AAᵀ) too, but it's simpler and more
#     numerically stable to derive U directly from V and σ using the formula above.
#
# PARAMETERS:
#     A : numpy 2D array — the matrix to decompose (any shape, e.g., 32×32 image block)
#     k : int — the target rank (how many singular values to keep)
#
# RETURNS:
#     U     : numpy 2D array of shape (m, k) — left singular vectors
#     sigma : numpy 1D array of shape (k,)   — singular values (descending order)
#     Vt    : numpy 2D array of shape (k, n) — right singular vectors (transposed)
#
# EDGE CASES TO HANDLE:
#     - If an eigenvalue is negative (numerical noise), clamp it to 0 before sqrt
#       HINT: eigenvalues = np.maximum(eigenvalues, 0)
#     - If a singular value is ~0, skip computing that column of U (avoid division by zero)
#       HINT: use a threshold like 1e-10
#
# EXAMPLE:
#     >>> A = np.random.rand(32, 32)            # A 32×32 image block
#     >>> U, sigma, Vt = custom_svd(A, k=8)     # Keep top 8 singular values
#     >>> A_compressed = U @ np.diag(sigma) @ Vt # Reconstruct the compressed block
#     >>> error = np.linalg.norm(A - A_compressed)
#     >>> print(f"Reconstruction error: {error}")  # Should be small but nonzero
#
# NOTEBOOK TEST:
#     Compare your output with NumPy's built-in:
#     >>> U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
#     >>> print("Our singular values:", sigma)
#     >>> print("NumPy singular values:", s_np[:k])
#     They should be very close (within ~0.001)
#
def custom_svd(A, k):
    """Compute rank-k SVD decomposition of matrix A from scratch."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Compute AᵀA (the covariance-like matrix)
    # HINT: AtA = np.dot(A.T, A)   — shape will be (n, n)
    
    # Step 2: Find the top-k eigenvalues and eigenvectors of AᵀA
    # HINT: eigenvalues, V = eigen_decompose(AtA, k)
    # NOTE: V's columns are the right singular vectors
    
    # Step 3: Handle numerical noise — clamp negative eigenvalues to 0
    # HINT: eigenvalues = np.maximum(eigenvalues, 0)
    
    # Step 4: Compute singular values (σ = √eigenvalue)
    # HINT: sigma = np.sqrt(eigenvalues)
    
    # Step 5: Compute U (left singular vectors) column by column
    # Formula: u_i = (A · v_i) / σ_i
    # HINT: Create an empty matrix U of shape (m, k) where m = A.shape[0]
    # HINT: Loop over each column i (0 to k-1):
    #     if sigma[i] > 1e-10:   (avoid division by zero)
    #         U[:, i] = np.dot(A, V[:, i]) / sigma[i]
    
    # Step 6: Compute Vᵀ (transpose of V)
    # HINT: Vt = V.T
    
    # Step 7: Return U, sigma, Vt
    
    pass  # ← Remove this once you write your code
