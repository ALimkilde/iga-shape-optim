from nutils import *
import numpy as np
import scipy as sc
import splipy as sp
import math
import os

def pad_knot_vec(knots,p):
    nz = len(np.where(knots == 0)[0])
    no = len(np.where(knots == 1)[0])
    goal = p+1

    if nz < goal:
        print('Knot vec padded')
        zs = np.zeros(goal-nz)
        knots = np.concatenate((zs,knots),axis = 0)
    elif nz > goal:
        print('Knot vec unpadded')
        knots = knots[nz-goal:len(knots)]
    if no < goal:
        os = np.ones(goal-no)
        knots = np.concatenate((knots,os),axis = 0)
    elif no > goal:
        knots = knots[0:len(knots)-no+goal]

    return knots


# construct topology, geometry and basis
verts = numpy.linspace(0, 1, 9)
nelems = len(verts)-1

ns = function.Namespace()

patches = [[0,1,6,4],[1,2,4,5],[2,3,5,7],[4,5,6,7],[0,6,3,7]]
patchverts = [[0,0],[2,0],[3,0],[5,0],[2,1],[3,1],[1,2],[4,2]]

#domain, geom = mesh.rectilinear( [verts,verts] )
domain, geom = mesh.multipatch(patches,nelems,patchverts = patchverts)

ns.basis = domain.basis('spline', degree=2)
ns.x = geom

grad      = ns.basis.grad(ns.x)
outer     = function.outer(grad,grad)
integrand = outer.sum(-1)

integrand2 = function.outer(ns.basis,ns.basis)

# construct matrix
K = domain.integrate( integrand,
    geometry=ns.x, ischeme='gauss3')

M = domain.integrate(integrand2,
    geometry=ns.x, ischeme='gauss3')

# construct dirichlet boundary constraints
cons = domain.boundary.project(
    0, onto=ns.basis, geometry=ns.x, ischeme='gauss1')

x,I,J = matrix.parsecons(cons,None,None,K.shape)

# solve linear system
Mnp = M.toarray()[np.ix_(I,J)]
Knp = K.toarray()[np.ix_(I,J)]

evals,evecs = sc.linalg.eigh(Knp,Mnp)

for i in range(2):
    print(i,' : ',evals[i])
    w = np.zeros(K.shape[0])
    w[J] = np.real(evecs[:,i])
    w[np.invert(J)] = x[np.invert(J)]
    sol = ns.basis.dot(w)

    # plot solution
    points, colors = domain.elem_eval( [ ns.x, sol], ischeme='bezier9', separate=True )

    folder = 'DiskTest1'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with plot.PyPlot( '%s/efun' %folder, index=i ) as plt:
        plt.mesh(points, colors )
        plt.colorbar()
