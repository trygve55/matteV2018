import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as pl

g = -9.81 #gravity constant earth (negative)
L = 2.0 #length
w = 0.30 #width
d = 0.030 #depth
density = 480.0 #density of material
E = 1.3*10**10 #Young's modulus
I = (w*d**3.0)/12.0 #inertia around the center of mass
m = density*w*d #mass per meter of beam
f = m*g #downward force per meter of beam
p = 100.0 #kg/m
mp = 50 #mass of the person on the end
fl = 0.3 #lenght of foot of person
constant = f/(24 * E * I);
emach = np.finfo(float).eps;#1/2**(52);

def makeStructureMatrix(n):
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
    b = sp.array([((L/n)**4 /(E * I)) * f] * n)

    return b

def getB6(n):
    h = L/n
    b = getB(n)

    for x in range(n):
        b[x] -= ((h)**4 /(E * I)) * p*g*math.sin((x*h)*math.pi/L) #p*g*math.sin((x*h + h/2)*math.pi/L)
        
    return b


def getB7(n):
    h = L / n
    b = getB(n)

    for x in range(n):

        if (L - h*n <= fl):
            b[x] -= ((h) ** 4 / (E * I)) * g * mp/fl
    return b

def getY(x):
    return (f/(24*E*I))*x**2*(x**2 - 4*L*x + 6 * L**2)

def getY6(x):
    return (f/(24*E*I))*x**2*(x**2 - 4*L*x + 6 * L**2) - ((g*p*L)/(E*I*math.pi))*((L**3/math.pi**3)*math.sin(math.pi*x/L) - x**3/6 + L*x**2/2 - L**2*x/math.pi**2)

def oppg5(i):
    out = []

    for i in range(i):
        n = 10 << i
        
        A = makeStructureMatrix(n)
        b = getB(n)
        
        y = getY(L) - spsolve(A, b)[-1]
        out.append({"n" : n,"kondis" : kondisjonstall(A.toarray()), "yDiff" : y, "h" : L/n})

    return out

def oppg6b(i):
    out = []
    
    for i in range(i):
        n = 10 << i
        
        A = makeStructureMatrix(n)
        b = getB6(n)
        y = getY6(L) - spsolve(A, b)[-1]
        out.append({"n" : n,"kondis" : kondisjonstall(A.toarray()), "yDiff" : y, "h" : L/n})

    return out


def oppg7(i):
    out = []

    for i in range(i):
        n = 10 << i

        A = makeStructureMatrix(n)
        b = getB7(n)
        y = spsolve(A, b)#[-1]
        out.append({"n": n, "kondis": kondisjonstall(A.toarray()), "y": y, "h": L / n})

    return out


def kondisjonstall(A):
    return linalg.norm(A,np.inf)*linalg.norm(linalg.inv(A),np.inf);


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
m = n;
n = 10;
y_e = np.array([getY(i/n) for i in range(2, 21, 2)]);
print("4.c. y_e: ", y_e, "\n");
A_y = np.matmul(A.toarray(), y_e)*(10000/(L**4));# SPØRRE RIVTZ OM DETTE
print("4.c. Derivert: ", A_y, "\n");
vector = np.array([f/(E * I)] * n);
print("4.d. f/(EI): ", vector, "\n");
print("4.d. Differanse: ", A_y - vector, "\n");
forward = linalg.norm(vector - A_y, np.inf);# Dette er feilen for den fjedederiverte.
print("4.d. Fram. feil: ", forward, "\n");
rel = forward/linalg.norm(vector, np.inf);
print("4.d. Rel. Fram. feil: ", rel, "\n");
print("4.d. Feil forstørring: ", rel/emach);
print("4.d. Kondisjonstall: ", kondisjonstall(A.toarray()));
#Mangler å se på kondisjonstallet.

# Tull med oppg e:
print("4.e. Differanse: ", y - y_e, "\n");
print("4.e. Fram. feil: ", linalg.norm((y - y_e), 1), "\n"); # feilen er for å finne y!
print("4.e. Maskin epsilon: ", emach);
n = m;

print("Oppg. 5")
oppg5data = oppg5(9)
for e in oppg5data:
    print(e)

print("Oppg. 6b")
oppg6data = oppg6b(9)
for e in oppg6data:
    print(e)

print("Oppg. 6c")
arrayY5 = []
arrayX5 = []
for e in oppg5data:
    arrayX5.append(e["h"])
    arrayY5.append(e["yDiff"])
arrayY6 = []
arrayX6 = []
for e in oppg6data:
    arrayX6.append(e["h"])
    arrayY6.append(e["yDiff"])

pl.subplot(211)
pl.yscale('log')
pl.xscale('log')
pl.xlabel("h")
pl.ylabel("Feil")
pl.plot(arrayX5, np.fabs(arrayY5))
pl.subplot(212)
pl.yscale('log')
pl.xscale('log')
pl.xlabel("h")
pl.ylabel("Feil")
pl.plot(arrayX6, np.fabs(arrayY6))
pl.show()

print("Oppg. 6d")

arrayYKondEps = []
arrayYerror = []
for e in oppg5data:
    arrayYerror.append(L**2/e["n"]**2)
    arrayYKondEps.append(e["kondis"] * np.finfo(float).eps)
    
pl.subplot(111)
pl.yscale('log')
pl.xscale('log')
pl.ylabel("kondisjon")
pl.ylabel("h")
pl.plot(arrayX5,arrayYerror)
pl.plot(arrayX5,arrayYKondEps)
pl.plot(arrayX5, np.fabs(arrayY5))
pl.show()

print("Oppg. 7")

oppg7data = oppg7(7)
#for e in oppg7data:
    #print(e)

y7 = []
x7 = []
for i in range(oppg7data[6]["n"]):
    x7.append(oppg7data[6]["h"]*i)
y7 = oppg7data[6]["y"]

#pl.xlim(-0.1, 2.1)
pl.ylim(-1.0, 1.0)
pl.gca().set_aspect('equal', adjustable='box')
pl.plot(x7, -y7)
pl.show()
