import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy import linalg
import math

g = -9.81 #gravity constant earth (negative)
L = 2.0 #length
w = 0.30 #width
d = 0.03 #depth
density = 480.0 #density of material
E = 1.3*10.0**10 #Young's modulus
I = (w*d**3.0)/12.0 #inertia around the center of mass
m = density*w*d #mass per meter of beam
f = m*g #downward force per meter of beam
p = 100.0 #kg/m

def makeStructureMatrix( n ):
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e,-4*e, e],[-2,-1,0,1,2],n,n)
    A = lil_matrix(A)
    B = csr_matrix([[16,-9,8/3,-1/4],
                    [16/17,-60/17,72/17,-28/17],
                    [-12/17,96/17,-156/17,72/17]])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]
    return csr_matrix(A)

def getB(n):
    h = L/n

    b = sp.array([(h**4 /(E * I)) * f] * n)

    return b

def getBextraWeight(n):
    h = L/n

    b = sp.array([(h**4/(E*I))*f] * n)

    for x in range(n):
        b[x] += sinusformethaug(x*h, h)

    return b

def getY(x):
    return (f/(24 * E * I))* x * x *(x * (x - 4 * L) + 6 * L * L)

#def numDeriv4(n):
#    h = L/n;
#    return (getY(n - 2*h) - 4 * (getY(n - h) + getY(n + h)) + 6 * getY(n) + getY(n + 2*h))/h**4;

def getYextraWeight(x):
    return (f/(24*E*I))*x**2*(x**2 - 4*L*x + 6 * L**2) #- ((g*p*l)/(E*I*math.pi))*((L**3/math.pi**3)*math.sin(math.pi*x/L) - x**3/6 + L*x**2/2) - L**2*x/math.pi**2)

def oppg5(i):
    out = []
    
    for i in range(i - 1):
        n = 20 * 2**i
        
        A = makeStructureMatrix(n)
        b = getB(n)
        
        y = getY(L) - spsolve(A, b)[-1]
        out.append({"n" : n,"kondis" : kondisjonstall(A.toarray()), "y" : y})

    return out

def oppg6b(i):
    out = []
    
    for i in range(i - 1):
        n = 20 * 2**i
        
        A = makeStructureMatrix(n)
        b = getBextraWeight(n)
        print(spsolve(A, b)[-1])
        print(b)
        print(getYextraWeight(n))
        y = getYextraWeight(L) - spsolve(A, b)[-1]
        out.append({"n" : n,"kondis" : kondisjonstall(A.toarray()), "y" : y})

    return out


def kondisjonstall(A):
    invers=linalg.inv(A)
    return linalg.norm(A,np.inf)*linalg.norm(invers,np.inf)

def sinusformethaug(x, h):
    return -h*p*g*math.sin((x + h/2)*math.pi/L)

n = 10
A = makeStructureMatrix(n)
b = getB(n)
print("Oppg. 2")
print(A.toarray())
print(b)

print("Oppg. 3") 
y = spsolve(A, b)
print(y)

print("Oppg. 4");
y_e = sp.array([getY(i/10) for i in range(2, 21, 2)]);
print("Y_e: ", y_e);
print(csr_matrix.multiply(A, y_e)*(1000/((L)**4)));

print("Oppg. 5")
for e in oppg5(6):
    print(e)

print("Oppg. 6b")
for e in oppg6b(6):
    print(e)

print("Oppg. 7")

