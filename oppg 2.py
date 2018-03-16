import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

def makeStructureMatrix( n ):
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e,-4*e, e],[-2,-1,0,1,2],n,n)
    A = lil_matrix(A)
    B = csr_matrix([[16.0,-9.0,8.0/3.0,1.0/4.0],
                    [16.0/17.0,-60.0/17.0,72.0/17.0,-28.0/17],
                    [-12.0/17.0,96.0/17,-156.0/17.0,72.0/17.0]])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]
    return A

print(makeStructureMatrix(10))
