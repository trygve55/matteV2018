import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as pl
import time

g = -9.81 #gravity constant earth (negative)
L = 2. #length
w = 0.3 #width
d = 0.03 #depth
density = 480. #density of material
E = 1.3*10**10 #Young's modulus
I = (w*d**3.)/12. #inertia around the center of mass
m = density*w*d #mass per meter of beam
f = m*g #downward force per meter of beam
p = 100. #kg/m
mp = 50 #mass of the person on the end
fl = 0.3 #lenght of foot of person
EI = E * I;
gp = p * g;
LL = L * L;
constantEI = f/(24 * EI);
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
    b = sp.array([((L/n)**4 * f)/EI] * n)

    return b

def getB6(n):
    c0 = (L/n)**4 * gp;
    c1 = math.pi/n
    b = getB(n)

    for x in range(n):
        b[x] -= (c0 * np.sin((x + 1)*c1))/EI #p*g*math.sin((x*h + h/2)*math.pi/L)
        
    return b


def getB7(n):
    h = L / n
    c = (g * mp * h**4)/(fl * EI);
    b = getB(n)

    for x in range(n):
        if (L - h*(x+1) <= fl):
            b[x] -= c;
    return b

def getY(x):
    return constantEI * x * x * (x * (x - 4*L) + 6 * LL)

def getY6(x):
    return constantEI * x * x * (x * (x - 4*L) + 6 * LL) - (gp*L)/(EI*math.pi)*(LL*(L*math.sin(math.pi*x/L) - math.pi)/(math.pi*math.pi*math.pi) - x*(x*(x + 3*L)/6))

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
        y = spsolve(A, b)[-1]
        yDiff = getY6(L) - y
        out.append({"n" : n,"kondis" : kondisjonstall(A.toarray()),"y" : y, "yDiff" : yDiff, "h" : L/n})

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
    return linalg.norm(A, np.inf)*linalg.norm(linalg.inv(A), np.inf);

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
A_y = np.matmul(A.toarray(), y_e)*10000 / (LL*LL);
print("4.c. Derivert: ", A_y, "\n");
vector = np.array([constantEI * 24] * n);
print("4.d. f/(EI): ", vector, "\n");
print("4.d. Differanse: ", A_y - vector, "\n");
forward = linalg.norm(vector - A_y, np.inf);# Dette er feilen for den fjedederiverte.
print("4.d. Fram. feil: ", forward, "\n");
rel = forward/linalg.norm(vector, np.inf);
print("4.d. Rel. Fram. feil: ", rel, "\n");
print("4.d. Feil forstørring: ", rel/emach);
print("4.d. Kondisjonstall: ", kondisjonstall(A.toarray()));
#Mangler å se på kondisjonstallet.

print("4.e. Differanse: ", y - y_e, "\n");
print("4.e. Fram. feil: ", linalg.norm((y - y_e), 1), "\n"); # feilen er for å finne y!
print("4.e. Maskin epsilon: ", emach);
n = m;

#utregninger for oppg 5, 6, 7
startTime = time.time()
oppg5data = oppg5(12)
print("Oppg 5 tid: ", time.time()-startTime, "s")
startTime = time.time()
oppg6data = oppg6b(12)
print("Oppg 6 tid: ", time.time()-startTime, "s")
startTime = time.time()
oppg7data = oppg7(8)
print("Oppg 7 tid: ", time.time()-startTime, "s")
startTime = time.time()

print("Oppg. 5")
arrayY5 = []
arrayX5 = []
for e in oppg5data:
    print(e)
    arrayX5.append(e["h"])
    arrayY5.append(e["yDiff"])

pl.yscale('log')
pl.xscale('log')
pl.xlabel("h")
pl.ylabel("Feil")
pl.title("Oppg. 5")
pl.plot(arrayX5, np.fabs(arrayY5), label="y feil")
pl.legend()
pl.show()


print("Oppg. 6b")
arrayY6 = []
arrayX6 = []
arrayY6y = []
for e in oppg6data:
    print(e)
    arrayX6.append(e["n"])
    arrayY6.append(e["yDiff"])
    arrayY6y.append(e["y"])

pl.xlabel("n")
pl.ylabel("y")
pl.xscale("log")
pl.title("Oppg. 6b")
pl.plot(arrayX6, np.fabs(arrayY6y), label="y")
pl.plot(arrayX6, np.fabs([getY6(L)]*arrayX6.__len__()), label="y eksakt")
pl.legend()
pl.show()

print("Oppg. 6c")

pl.yscale('log')
pl.xscale('log')
pl.xlabel("n")
pl.ylabel("Feil")
pl.title("Oppg. 6c")
pl.plot(arrayX6, np.fabs(arrayY6), label="y feil")
pl.legend()
pl.show()

print("Oppg. 6d")

arrayYKondEps = []
arrayYerror = []
for e in oppg5data:
    arrayYerror.append(LL/e["n"]**2)
    arrayYKondEps.append(e["kondis"] * np.finfo(float).eps)
    
pl.subplot(111)
pl.yscale('log')
pl.xscale('log')
pl.ylabel("Verdi")
pl.xlabel("h")
pl.title("Oppg. 6d")
pl.plot(arrayX5,arrayYerror, label="L^2/n^2")
pl.plot(arrayX5,arrayYKondEps, label="C * machine epsilon")
pl.plot(arrayX5, np.fabs(arrayY6), label="feil 6c")
pl.legend()
pl.show()


print("Oppg. 6f")
lowYDiff = np.infty
lowYDiffN = 1
for e in oppg6data:
    if (np.fabs(e["yDiff"]) < lowYDiff):
        lowYDiff = np.fabs(e["yDiff"])
        lowYDiffN = e["n"]
print("Den optimale verdien av n er " + str(lowYDiffN) + ", fordi dette gir lavest feil. Dette er fordi h^4 blir nære marskinepsilon og derfor blir mer unøyaktig etter dette punktet(feil?)")

print("Oppg. 7")

#for e in oppg7data:
    #print(e)
print("Stupebrettet bøyes ned " + str(oppg7data[3]["y"][-1]) + " m.")

y7 = []
x7 = []
for i in range(oppg7data[6]["n"]):
    x7.append(oppg7data[6]["h"]*i)
y7 = oppg7data[6]["y"]

#pl.xlim(-0.1, 2.1)
pl.ylim(-1.0, 1.0)
pl.gca().set_aspect('equal', adjustable='box')
pl.plot(x7, -y7, label='Stupebrett')
pl.title('Oppg. 7')
pl.legend()
pl.show()

print("Se oppg 6f for n.")
