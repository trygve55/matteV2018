import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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
    return csr_matrix(A)

def getB(n):
    g = -9.81 #gravity constant earth (negative)
    L = 2.0 #length
    w = 0.30 #width
    d = 0.03 #depth
    density = 480.0 #density of material

    E = 1.3*10.0**10 #Young's modulus
    I = (w*d**3.0)/12.0 #inertia around the center of mass
    m = density*w*d #mass per meter of beam
    f = m*g #downward force per meter of beam
    h = L/n

    print(f)

    b = [(h**4/(E*I))*f] *n

    return b

n = 10
A = makeStructureMatrix(n)
b = getB(n)
print(A.toarray())
print(b)
#b = 
y = spsolve(A, b)
print(y)

for i in range(11):
    n = 20 * 2**i

    A = makeStructureMatrix(n)
    b = getB(n)
    print(A.toarray())
    print(b)
    #b = 
    y = spsolve(A, b)
