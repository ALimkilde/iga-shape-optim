import splipy as sp
import splipy.surface_factory as SurfaceFactory
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import scipy.linalg as LinAlg
from timeit import default_timer as timer

# Helper functions
def generate_knot_vec(mid,p):
    zs = np.zeros(p+1)
    os = np.ones(p+1)
    return np.concatenate((zs,mid,os),axis = 0)

def rescale_gauss_points(t,a,b):
    return (b*(t+1) - a*(t-1))/2.0

def generate_all_gauss_points(t,w,knots):
    k_u = np.unique(knots)

    n_g = len(t)
    n_i = len(k_u)-1

    T = np.empty(n_g*n_i)
    W = np.empty(n_g*n_i)

    for i in range(0,n_i):
        a = k_u[i]
        b = k_u[i+1]
        t_r = rescale_gauss_points(t,a,b)
        T[i*n_g:i*n_g+n_g] = t_r
        W[i*n_g:i*n_g+n_g] = w*(b-a)/2

    return T,W

def get_2d_grid(x,y):
    n_x = len(x)
    n_y = len(y)
    X = np.kron(x,np.ones(n_y))
    Y = np.kron(np.ones(n_x),y)
    return X,Y

def evaluate_2d_basis(basis_x,basis_y,x,y,dx = 0, dy = 0):
    Bx = basis_x.evaluate(x, d = dx)
    By = basis_y.evaluate(y, d = dy)

    return np.kron(Bx,By)

def pad_knot_vec(knots,p):
    nz = len(np.where(knots == 0)[0])
    no = len(np.where(knots == 1)[0])
    goal = p+1

    if nz < goal:
        #print('Knot vec padded')
        zs = np.zeros(goal-nz)
        knots = np.concatenate((zs,knots),axis = 0)
    elif nz > goal:
        #print('Knot vec unpadded')
        knots = knots[nz-goal:len(knots)]
    if no < goal:
        os = np.ones(goal-no)
        knots = np.concatenate((knots,os),axis = 0)
    elif no > goal:
        knots = knots[0:len(knots)-no+goal]

    return knots

def get_greville(p,knots):
    n = len(knots) - p - 1
    grev = np.empty(n)
    iter = 0
    for i in range(1,n+1):
        print(knots[i:i+p])
        grev[iter] = np.average(knots[i:i+p])
        iter += 1
    return grev

def GordonHall(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y):
    nx = len(p1x)
    ny = len(q1x)
    xi = np.linspace(0,1,nx)
    eta = np.linspace(0,1,ny)

    Xi,Eta = np.meshgrid(xi,eta)

    xAx = np.zeros((nx,ny))
    xAy = np.zeros((nx,ny))
    xBx = np.zeros((nx,ny))
    xBy = np.zeros((nx,ny))
    xCx = np.zeros((nx,ny))
    xCy = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            xAx[i,j] = (1-xi[i])*q1x[j] + xi[i]*q2x[j]
            xAy[i,j] = (1-xi[i])*q1y[j] + xi[i]*q2y[j]
            xBx[i,j] = (1-eta[j])*p1x[i] + eta[j]*p2x[i]
            xBy[i,j] = (1-eta[j])*p1y[i] + eta[j]*p2y[i]
            xCx[i,j] = (1-xi[i])*(1-eta[j])*p1x[0] + xi[i]*(1-eta[j])*p1x[-1] + (1-xi[i])*eta[j]*p2x[0] + xi[i]*eta[j]*p2x[-1]
            xCy[i,j] = (1-xi[i])*(1-eta[j])*p1y[0] + xi[i]*(1-eta[j])*p1y[-1] + (1-xi[i])*eta[j]*p2y[0] + xi[i]*eta[j]*p2y[-1]


    return xAx + xBx - xCx, xAy + xBy - xCy

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def blocktridiag(A,m):
    n,m = A.shape
    M1 = np.kron(np.eye(m),A)
    v = np.ones(n*(n-1))
    return M1 + np.diag(v,-n) + np.diag(v,n)

def mul(a,b):
    return np.multiply(a,b)

def SpringModel(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y):
    nx = len(p1x)
    ny = len(q1x)
    d = -4*np.ones(nx)
    a = np.ones(nx-1)
    B = tridiag(a,d,a)
    S = blocktridiag(B,ny)

    fx = np.zeros(nx)
    fx[0] = 1
    fx[nx-1] = 1
    fy = np.zeros(ny)
    fy[0] = 1
    fy[nx-1] = 1

    Fx,Fy = get_2d_grid(fx,fy)
    Fixed_x = np.argwhere(Fx == 1)
    Fixed_y = np.argwhere(Fy == 1)
    Fixed = np.union1d(Fixed_x,Fixed_y)

    NotFixed = np.setdiff1d(list(range(nx*ny)),Fixed)

    Anf = S[:,NotFixed]
    Af  = S[:,Fixed]

    Cx,Cy = GordonHall(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y)
    Cx = Cx.reshape(nx*ny)
    Cy = Cy.reshape(nx*ny)

    rhsx = -np.matmul(Af[NotFixed,:],Cx[Fixed])
    rhsy = -np.matmul(Af[NotFixed,:],Cy[Fixed])

    Cx[NotFixed],Cy[NotFixed] = np.linalg.solve(Anf[NotFixed,:],rhsx),np.linalg.solve(Anf[NotFixed,:],rhsy)
    return Cx,Cy

def plot_2D_surface(surface):
    u = np.linspace(surface.start('u'), surface.end('u'), 30)
    v = np.linspace(surface.start('v'), surface.end('v'), 30)
    x = surface(u,v)
    plt.plot(x[:,:,0],   x[:,:,1],   'k-')
    plt.plot(x[:,:,0].T, x[:,:,1].T, 'k-')
    plt.axis('equal')
    plt.show()

def plot_2D_curve(curve, show_controlpoints=False,col = 'b'):
    t = np.linspace(0, 1, 150)
    x = curve(t)
    plt.plot(x[:,0], x[:,1],col)
    if(show_controlpoints):
        plt.plot(curve[:,0], curve[:,1], 'rs-')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.axis('equal')
    #plt.show()

def project1D(basis,fun,degree=10):
    T,W = basis.getGaussPoints(deg = degree)
    M = basis.generateMassMatrix()
    B = basis.evaluate(T)
    Fun = fun(T)
    rhs = B.T.dot(np.diag(W)).dot(Fun)

    return LinAlg.solve(M,rhs.T)

def plotOut(out,bas):
    cols = ['b','r','g','k']
    plt.figure(2)
    for cps,col in zip(out,cols):
        c = sp.Curve(basis = bas, controlpoints = cps)
        plot_2D_curve(c,show_controlpoints = False,col = col)

def generateSplineCircle(basis1,basis2):
    theta = np.array([1,3,5,7,9])*math.pi/4

    out = ()

    for i in [2,0,1,3]:
        e = theta[i+1]
        s = theta[i]
        t = e-s

        fun = lambda x: np.sin(t*x-s)
        fun2 = lambda x: np.cos(t*x-s)
        if i == 1 or i == 3:
            bas = basis2
        else:
            bas = basis1
        c1 = project1D(bas,fun)
        c2 = project1D(bas,fun2)
        if i == 2 or i == 3:
            c1 = c1[::-1]
            c2 = c2[::-1]
        cps = np.zeros((bas.n,2))
        cps[:,0] = c1.ravel()
        cps[:,1] = c2.ravel()

        out += (cps,)

    return out

def generateSplineLine(basis,a,b):
    fun1 = lambda x: x*a[0] + (1-x)*b[0]
    fun2 = lambda x: x*a[1] + (1-x)*b[1]
    c1 = project1D(basis,fun1)
    c2 = project1D(basis,fun2)
    cps = np.zeros((basis.n,2))
    cps[:,0] = c1.ravel()
    cps[:,1] = c2.ravel()
    return cps

def generateTopP(basis1,basis2,bottom,p1 = np.array([-8,-2]),p2 = np.array([8,-2])):
    out = (bottom,)

    t = math.pi
    d = p2 - p1
    r = math.sqrt(d[0]**2 + d[1]**2)/2
    m = 1/2*(p1 + p2)
    fun1 = lambda x: m[0] + np.sin(t*x - math.pi/2)*r
    fun2 = lambda x: m[1] + np.cos(t*x - math.pi/2)*r
    c1 = project1D(basis1,fun1)
    c2 = project1D(basis1,fun2)
    #c1 = c1[::-1]
    #c2 = c2[::-1]
    cps = np.zeros((basis1.n,2))
    cps[:,0] = c1.ravel()
    cps[:,1] = c2.ravel()
    out += (cps,)

    q1 = bottom[0,:]
    q2 = bottom[-1,:]
    cps1 = generateSplineLine(basis2,p1,q1)
    cps2 = generateSplineLine(basis2,q2,p2)
    cps2 = cps2[::-1,:]
    out += (cps1,cps2,)

    return out

def generateLeftP(basis1,basis2,left,top,p = np.array([-1,-2])):
    out = ()

    cps1 = generateSplineLine(basis1,left[-1,:],p)
    out += (cps1,top[::-1,:],left[::-1,:],)

    cps2 = generateSplineLine(basis2,p,top[0,:])
    out += (cps2[::-1,:],)
    return out

def generateRightP(basis1,basis2,right,top,p = np.array([1,-2])):
    out = ()

    cps1 = generateSplineLine(basis1,p,right[-1,:])
    out += (cps1,top,)

    cps2 = generateSplineLine(basis2,p,top[0,:])
    out += (cps2[::-1,:],right[::-1,:])
    return out

def generateBottomP(basis1,basis2,left,top,right):
    cps1 = generateSplineLine(basis1,left[0,:],right[0,:])
    out = (cps1,top,left,right)

    return out

def generateParametrizationOfDisc(p,q,knots = None):
    if knots == None:
        knots = generate_knot_vec(np.linspace(0,1,4),p)
        knots = pad_knot_vec(knots,p)
        print(knots)

    n = len(knots) - p - 1

    P = Parametrization(p,q,knots,knots,[],[])
    bas1 = P.basis_xi
    bas2 = P.basis_eta

    CpsM = generateSplineCircle(bas1,bas2)

    P.getParametrization(*CpsM,method = 'GordonHall')
    plt.figure(1)
    P.plot()

    CpsT = generateTopP(bas1,bas2,CpsM[1])

    P.getParametrization(*CpsT,method = 'GordonHall')
    plt.figure(1)
    P.plot()

    CpsL = generateLeftP(bas1,bas2,CpsT[2],CpsM[2])

    P.getParametrization(*CpsL,method = 'Spring')
    plt.figure(1)
    P.plot()

    CpsR = generateRightP(bas1,bas2,CpsT[3],CpsM[3])

    P.getParametrization(*CpsR,method = 'Spring')
    plt.figure(1)
    P.plot()

    CpsB = generateBottomP(bas1,bas2,CpsL[3],CpsM[0],CpsR[2])

    P.getParametrization(*CpsB,method = 'Spring')
    plt.figure(1)
    P.plot()
    plt.show()
    plt.show()


# Classes
class Basis(sp.BSplineBasis):
    def __init__(self,p,knots):
        sp.BSplineBasis.__init__(self,order = p+1, knots = knots)
        self.p = p
        self.knots = knots
        self.n = len(knots) - p - 1
    def plotBasis(self,t = np.linspace(0,1,500)):
        B = self.evaluate(t)
        plt.plot(t,B)
        plt.show()
    def getGaussPoints(self,deg = 0):
        if deg == 0:
            deg = self.p + 1
        t_xi,w_xi = np.polynomial.legendre.leggauss(deg)
        T_xi,W_xi = generate_all_gauss_points(t_xi,w_xi,self.knots)
        return T_xi,W_xi
    def generateMassMatrix(self):
        T,W = self.getGaussPoints()
        B = self.evaluate(T)
        return B.T.dot(np.diag(W)).dot(B)

class Parametrization:
    def __init__(self,p,q,knots_xi,knots_eta,Cx,Cy):
         self.basis_xi = Basis(p,knots_xi)
         self.basis_eta = Basis(q,knots_eta)
         self.Cx = Cx
         self.Cy = Cy

    ## General methods
    def map(self,xi,eta):
         R = evaluate_2d_basis(self.basis_xi,self.basis_eta,xi,eta)
         x = np.matmul(self.Cx,np.transpose(R))
         y = np.matmul(self.Cy,np.transpose(R))
         return x,y
    def plot(self,x = None):
        if not x is None:
            nnfx = len(self.NotFixed_x)
            nnfy = len(self.NotFixed_y)
            self.Cx[self.NotFixed_x] = x[0:nnfx]
            self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]

        if  not hasattr(self,'M_dj'):
            self.generateMassMatrix()

        print('mindvec = ',np.min(self.getDVec()))

        for xi in self.basis_xi.knots:
            eta = np.linspace(0,1,100)

            x,y = self.map(xi,eta)
            n,m = x.shape
            xtmp = np.zeros(m)
            ytmp = np.zeros(m)
            xtmp[:] = x
            ytmp[:] = y
            plt.plot(xtmp,ytmp,'k-')

        for eta in self.basis_eta.knots:
            xi = np.linspace(0,1,100)

            x,y = self.map(xi,eta)
            n,m = x.shape
            xtmp = np.zeros(m)
            ytmp = np.zeros(m)
            xtmp[:] = x
            ytmp[:] = y
            plt.plot(xtmp,ytmp,'k-')
        plt.plot(self.Cx,self.Cy,'bo')
        #plt.xlim([-0.1,1.1])
        #plt.ylim([-0.1,1.1])
        plt.axis('equal')
        #plt.show()

    ## Loading fixing updating etc...
    def saveCxy(self,fname = 'Cxy'):
        sav = np.zeros((len(self.Cx),2))
        sav[:,0] = self.Cx
        sav[:,1] = self.Cy

        np.savetxt(fname,sav)
    def loadCxy(self,fname = 'Cxy'):
        mat = np.loadtxt(fname)
        self.Cx = mat[:,0]
        self.Cy = mat[:,1]
    def fix(self,Fixed_x,Fixed_y):
        N = self.basis_xi.n*self.basis_eta.n;
        self.NotFixed_x = np.setdiff1d(list(range(N)),Fixed_x)
        self.NotFixed_y = np.setdiff1d(list(range(N)),Fixed_y)
        self.NotFixed = np.concatenate((self.NotFixed_x,self.NotFixed_y+N),axis = 0)
    def getCxy(self):
        return self.Cx,self.Cy
    def getNonFixedCxy(self):
        return self.Cx[self.NotFixed_x],self.Cy[self.NotFixed_y]
    def update(self,Cx,Cy):
        self.Cx = Cx
        self.Cy = Cy

    ## Get spline coefficients for Determinant J
    def generateSlineBasisForDetJ(self):
        p_dj = 2*self.basis_xi.p-1
        q_dj = 2*self.basis_xi.p-1

        mult_xi_dj = self.basis_xi.p + 1
        mult_eta_dj = self.basis_xi.p + 1

        knots_dj_xi = np.sort(np.repeat(np.unique(self.basis_xi.knots),mult_xi_dj),axis = 0)
        knots_dj_eta = np.sort(np.repeat(np.unique(self.basis_xi.knots),mult_eta_dj),axis = 0)

        knots_dj_xi = pad_knot_vec(knots_dj_xi,p_dj)
        knots_dj_eta = pad_knot_vec(knots_dj_eta,q_dj)

        self.basis_dj_xi = Basis(p_dj,knots_dj_xi)
        self.basis_dj_eta = Basis(q_dj,knots_dj_eta)

        self.T_xi,self.W_xi = self.basis_dj_xi.getGaussPoints(deg = 2*self.basis_xi.p + 1)
        self.T_eta,self.W_eta = self.basis_dj_eta.getGaussPoints(deg = 2*self.basis_xi.p + 1)

        self.W = np.kron(self.W_xi,self.W_eta)

        self.B_dj_xi = self.basis_dj_xi.evaluate(self.T_xi)
        self.B_dj_eta = self.basis_dj_eta.evaluate(self.T_eta)

        self.R_dj = evaluate_2d_basis(self.basis_dj_xi,self.basis_dj_eta,self.T_xi,self.T_eta)

        self.d_xiR = evaluate_2d_basis(self.basis_xi,self.basis_eta,self.T_xi,self.T_eta,dx = 1)
        self.d_etaR = evaluate_2d_basis(self.basis_xi,self.basis_eta,self.T_xi,self.T_eta,dy = 1)
    def generateMassMatrix(self):
        self.generateSlineBasisForDetJ()

        M_1d_xi = np.transpose(self.B_dj_xi) * np.diag(self.W_xi) * self.B_dj_xi
        M_1d_eta = np.transpose(self.B_dj_eta) * np.diag(self.W_eta) * self.B_dj_eta

        self.M_dj = np.kron(M_1d_xi,M_1d_eta)
        self.lu_M_dj = LinAlg.lu_factor(self.M_dj)
    def getDeterminantJ(self,xi,eta):
        self.Cx_arr = self.Cx
        self.Cy_arr = self.Cy

        self.d_xiR = evaluate_2d_basis(self.basis_xi,self.basis_eta,xi,eta,dx = 1)
        self.d_etaR = evaluate_2d_basis(self.basis_xi,self.basis_eta,xi,eta,dy = 1)

        self.J1 = np.matmul(np.transpose(self.Cx_arr),np.transpose(self.d_xiR))
        self.J2 = np.matmul(np.transpose(self.Cy_arr),np.transpose(self.d_xiR))
        self.J3 = np.matmul(np.transpose(self.Cx_arr),np.transpose(self.d_etaR))
        self.J4 = np.matmul(np.transpose(self.Cy_arr),np.transpose(self.d_etaR))

        return np.multiply(self.J1,self.J4) - np.multiply(self.J3,self.J2)
    def getDeterminantJ2(self):
        self.Cx_arr = self.Cx
        self.Cy_arr = self.Cy

        self.J1 = np.matmul(np.transpose(self.Cx_arr),np.transpose(self.d_xiR))
        self.J2 = np.matmul(np.transpose(self.Cy_arr),np.transpose(self.d_xiR))
        self.J3 = np.matmul(np.transpose(self.Cx_arr),np.transpose(self.d_etaR))
        self.J4 = np.matmul(np.transpose(self.Cy_arr),np.transpose(self.d_etaR))

        return np.multiply(self.J1,self.J4) - np.multiply(self.J3,self.J2)

    def getDVec(self):
        detJ = self.getDeterminantJ2()

        d_tilde = self.R_dj.T.dot(mul(detJ,self.W).T)
        dtmp = LinAlg.lu_solve(self.lu_M_dj,d_tilde)
        #dtmp = np.linalg.solve(self.M_dj,d_tilde)
        return dtmp
    def getJacDVec(self):
        detJ = self.getDeterminantJ2()
        N_gauss = len(self.T_xi)*len(self.T_eta)
        N = self.basis_xi.n*self.basis_eta.n
        dDetJ_cx = np.empty((N_gauss,N))
        dDetJ_cy = np.empty((N_gauss,N))

        J1W = np.transpose(self.J1)
        J2W = np.transpose(self.J2)
        J3W = np.transpose(self.J3)
        J4W = np.transpose(self.J4)

        for i in range(N):
            Rxi = self.d_xiR[:,i]
            Reta = self.d_etaR[:,i]
            dDetJ_cx[:,i] = np.multiply(Rxi,J4W).ravel() - np.multiply(Reta,J2W).ravel()
            dDetJ_cy[:,i] = np.multiply(Reta,J1W).ravel() - np.multiply(Rxi,J3W).ravel()
            dDetJ_cx[:,i] = mul(dDetJ_cx[:,i],self.W)
            dDetJ_cy[:,i] = mul(dDetJ_cy[:,i],self.W)

        JacDVec_x = self.R_dj.T.dot(dDetJ_cx)
        JacDVec_y = self.R_dj.T.dot(dDetJ_cy)

        N_dj = self.basis_dj_xi.n*self.basis_dj_eta.n

        JacDVec = np.zeros((N_dj,2*N))
        JacDVec[:,0:N] = LinAlg.lu_solve(self.lu_M_dj,JacDVec_x)
        JacDVec[:,N:2*N] = LinAlg.lu_solve(self.lu_M_dj,JacDVec_y)

        return JacDVec
    def getJacDVecFixed(self):
        JacDVec = self.getJacDVec()
        return JacDVec[:,self.NotFixed]

    ## Get som differential geometry measures
    def getFirstFundamentalForm(self,xi,eta):
        self.getDeterminantJ(xi,eta)
        self.g11 = np.power(self.J1,2) + np.power(self.J2,2)
        self.g12 = np.multiply(self.J1,self.J3) + np.multiply(self.J2,self.J4)
        self.g22 = np.power(self.J3,2) + np.power(self.J4,2)
    def getLiao(self,x):
        self.updateCxyFromx(x)
        self.getFirstFundamentalForm(self.T_xi,self.T_eta)
        mL = np.power(self.g11,2) + np.power(self.g22,2) + 2*np.power(self.g12,2)
        liao = mL.dot(self.W)


        return float(liao[0,0])
    def getJacLiao(self,x):
        self.updateCxyFromx(x)
        self.getFirstFundamentalForm(self.T_xi,self.T_eta)
        h1x = mul(self.J1,self.g11) + mul(self.J3,self.g12)
        h2x = mul(self.J3,self.g22) + mul(self.J1,self.g12)
        Wh1x = mul(self.W,h1x)
        Wh2x = mul(self.W,h2x)
        dLdcx = 4*Wh1x.dot(self.d_xiR) + 4*Wh2x.dot(self.d_etaR)

        h1y = mul(self.J2,self.g11) + mul(self.J4,self.g12)
        h2y = mul(self.J4,self.g22) + mul(self.J2,self.g12)
        Wh1y = mul(self.W,h1y)
        Wh2y = mul(self.W,h2y)
        dLdcy = 4*Wh1y.dot(self.d_xiR) + 4*Wh2y.dot(self.d_etaR)

        self.dLdcx = np.transpose(dLdcx)
        self.dLdcy = np.transpose(dLdcy)

        dLdc = np.concatenate((self.dLdcx,self.dLdcy),axis = 0)
        out = np.zeros(len(dLdc[self.NotFixed]))
        out[:] = dLdc[self.NotFixed,0].ravel()
        return out
    def getLiaoAndHessian(self,x):
        # Method to return liao funct., differential and hessian of Liao.
        l = self.getLiao(x)
        dl = self.getJacLiao(x)
        d_xidc = self.d_xiR
        d_etadc = self.d_etaR

        xxi = self.J1.T
        xeta = self.J3.T
        yxi = self.J2.T
        yeta = self.J4.T

        xxi2 = np.power(xxi,2)
        xeta2 = np.power(xeta,2)
        yxi2 = np.power(yxi,2)
        yeta2 = np.power(yeta,2)

        dh1xdcx = mul(d_xidc,3*xxi2 + yxi2 + xeta2) + mul(d_etadc,2*mul(xxi,xeta) + mul(yxi,yeta))
        dh2xdcx = mul(d_etadc,3*xeta2 + yeta2 + xxi2) + mul(d_xidc,2*mul(xeta,xxi) + mul(yxi,yeta))

        Hxx = 4*d_xidc.T.dot(mul(self.W[:,None],dh1xdcx)) + 4*d_etadc.T.dot(mul(self.W[:,None],dh2xdcx))

        dh1ydcx = mul(d_xidc,xxi2 + 3*yxi2 + yeta2) + mul(d_etadc,mul(xxi,xeta) + 2*mul(yxi,yeta))
        dh2ydcx = mul(d_etadc,xeta2 + 3*yeta2 + yxi2) + mul(d_xidc,mul(xeta,xxi) + 2*mul(yxi,yeta))

        Hyy = 4*d_xidc.T.dot(mul(self.W[:,None],dh1ydcx)) + 4*d_etadc.T.dot(mul(self.W[:,None],dh2ydcx))

        dh1xdcy = mul(d_xidc,2*mul(xxi,yxi) + mul(xeta,yeta)) + mul(d_etadc,mul(xeta,yxi))
        dh2xdcy = mul(d_etadc,2*mul(xeta,yeta) + mul(xxi,yxi)) + mul(d_xidc,mul(yeta,xxi))

        Hxy = 4*d_xidc.T.dot(mul(self.W[:,None],dh1xdcy)) + 4*d_etadc.T.dot(mul(self.W[:,None],dh2xdcy))

        n,m = Hxx.shape
        H = np.zeros((2*n,2*n))
        H[0:n,0:n] = Hxx
        H[n:2*n,n:2*n] = Hyy
        H[0:n,n:2*n] = Hxy
        H[n:2*n,0:n] = Hxy.T

        NFx,NFy = np.meshgrid(self.NotFixed,self.NotFixed)

        return l, dl, H[NFx,NFy]

    ## Parametrization method by maximizing the determinant
    def updateCxyFromx(self,x):
        nnfx = len(self.NotFixed_x)
        nnfy = len(self.NotFixed_y)
        self.Cx[self.NotFixed_x] = x[0:nnfx]
        self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]
    def const(self,x):
        nnfx = len(self.NotFixed_x)
        nnfy = len(self.NotFixed_y)
        self.Cx[self.NotFixed_x] = x[0:nnfx]
        self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]
        d = self.getDVec()
        out = np.zeros(len(d))
        out[:] = d.ravel() - x[-1]

        return np.asarray(out.ravel())
    def constjac(self,x):
        nnfx = len(self.NotFixed_x)
        nnfy = len(self.NotFixed_y)
        self.Cx[self.NotFixed_x] = x[0:nnfx]
        self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]
        jacdvec = self.getJacDVecFixed()
        n = jacdvec.shape[0]
        m = jacdvec.shape[1]
        out = np.zeros((n,m+1))
        out[0:n,0:m] = jacdvec
        out[:,m] = -1
        return out
    def loadz(self,fname = 'z',delta = 0):
        self.dz = np.loadtxt(fname)*delta
    def getxLiao(self):
        fx = np.zeros(n)
        fx[0] = 1
        fx[n-1] = 1
        fy = np.zeros(n)
        fy[0] = 1
        fy[n-1] = 1

        Fx,Fy = get_2d_grid(fx,fy)
        Fixed_x = np.argwhere(Fx == 1)
        Fixed_y = np.argwhere(Fy == 1)
        Fixed = np.union1d(Fixed_x,Fixed_y)

        self.fix(Fixed,Fixed)
        nc = n*n
        nf = len(Fixed)
        nnf = nc-nf
        #
        Cxnf,Cynf = self.getNonFixedCxy()
        x = np.zeros(nnf*2)
        x[0:nnf] = Cxnf
        x[nnf:nnf*2] = Cynf
        return x
    def getParametrization(self,p1,p2,q1,q2,method = 'MaxDet'):
        p1x = p1[:,0]
        p1y = p1[:,1]
        p2x = p2[:,0]
        p2y = p2[:,1]
        q1x = q1[:,0]
        q1y = q1[:,1]
        q2x = q2[:,0]
        q2y = q2[:,1]
        self.Cx,self.Cy = SpringModel(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y)

        self.generateMassMatrix()
        d = self.getDVec()
        dmin = np.min(d)
        #self.plot()
        #
        if method == 'Spring':
            return
        if method == 'GordonHall':
            self.Cx,self.Cy = GordonHall(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y)
            self.Cx = self.Cx.reshape((self.basis_xi.n*self.basis_eta.n))
            self.Cy = self.Cy.reshape((self.basis_xi.n*self.basis_eta.n))
        if method == 'MaxDet' or method == 'Liao':
            fx = np.zeros(n)
            fx[0] = 1
            fx[n-1] = 1
            fy = np.zeros(n)
            fy[0] = 1
            fy[n-1] = 1

            Fx,Fy = get_2d_grid(fx,fy)
            Fixed_x = np.argwhere(Fx == 1)
            Fixed_y = np.argwhere(Fy == 1)
            Fixed = np.union1d(Fixed_x,Fixed_y)

            self.fix(Fixed,Fixed)

            # JacDVec = P.getJacDVecFixed()
            #
            nc = n*n
            nf = len(Fixed)
            nnf = nc-nf
            #
            Cxnf,Cynf = self.getNonFixedCxy()
            x = np.zeros(nnf*2)
            x[0:nnf] = Cxnf
            x[nnf:nnf*2] = Cynf
        if method == 'MaxDet':
            print('Dmin : ', dmin)
            x0 = np.concatenate((x,np.array([dmin])),axis = 0)
            #
            jacmat = np.zeros(2*nnf+1)
            jacmat[-1] = -1
            #
            objfun = lambda x: -x[-1]
            Jacobj = lambda x: jacmat
            #
            #
            const = {'type' : 'ineq', 'fun' : self.const,'jac': self.constjac}
            opts = {'maxiter': 2000, 'disp': True}
            out = minimize(objfun,x0,jac = Jacobj,constraints = const,options = opts,method = 'SLSQP')
            print('z = ', out.x[-1])
            if out.x[-1] < 0:
                print('WARNING : maximization of determinant did not yield valid parametrization.')
            self.plot(x = out.x)
            #self.saveCxy('Jigsaw2')
            #np.savetxt('Jigsaw2_z',np.array([out.x[-1]]))
        if method == 'Liao':
            if not hasattr(self,'dz'):
                self.dz = 0
            x0 = x
            #
            const = {'type' : 'ineq', 'fun' : P.const2,'jac': P.constjac2}
            opts = {'maxiter': 2000, 'disp': True}

            out = minimize(P.getLiao,x0,jac = P.getJacLiao,constraints = const,options = opts)

            print(' = ', out.fun)
            print('N.o. iterations : ', out.nit)
            P.plot(x = out.x)
            P.saveCxy('Jigsaw2_Liao')
        if method == 'Poisson':
            print('Parametrization by the Poisson')
            p1 = np.zeros((10,2)); p1[:,0] = p1x; p1[:,1] = p1y;
            p2 = np.zeros((10,2)); p2[:,0] = p2x; p2[:,1] = p2y;
            q1 = np.zeros((10,2)); q1[:,0] = q1x; q1[:,1] = q1y;
            q2 = np.zeros((10,2)); q2[:,0] = q2x; q2[:,1] = q2y;

            curves = []
            cpoints = [p1,q2,p2,q1]
            bs = [self.basis_xi, self.basis_eta, self.basis_xi,self.basis_eta]
            for p in cpoints:
                c = sp.Curve(basis = bs.pop(0),controlpoints = p)
                curves.append(c)

            surface = SurfaceFactory.edge_curves(curves,type = 'poisson')
            cps = surface.controlpoints
            self.Cx = cps[:,:,0].ravel(); self.Cy = cps[:,:,1].ravel()
    ## Parametrization Method by minimizing the liao measure.
    def const2(self,x):
        nnfx = len(self.NotFixed_x)
        nnfy = len(self.NotFixed_y)
        self.Cx[self.NotFixed_x] = x[0:nnfx]
        self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]
        d = self.getDVec()
        out = np.zeros(len(d))
        out[:] = d.ravel() - self.dz

        return np.asarray(out)
    def constjac2(self,x):
        nnfx = len(self.NotFixed_x)
        nnfy = len(self.NotFixed_y)
        self.Cx[self.NotFixed_x] = x[0:nnfx]
        self.Cy[self.NotFixed_y] = x[nnfx:nnfx+nnfy]
        jacdvec = self.getJacDVecFixed()
        return jacdvec

    ## Test Stuff
    def testMassMatrix(self):
        R = evaluate_2d_basis(self.basis_dj_xi,self.basis_dj_eta,self.T_xi,self.T_eta)

        n_g_xi = len(self.T_xi)
        n_g_eta = len(self.T_eta)

        f = lambda x,y: x**5

        X,Y = get_2d_grid(self.T_xi,self.T_eta)

        f_vec = f(X,Y)
        self.W = np.kron(self.W_xi,self.W_eta)

        F = np.matmul(np.transpose(R),np.multiply(f_vec,self.W))

        u = np.linalg.solve(self.M_dj,np.transpose(F))

        u_vec = np.matmul(R,u)

        print(np.max(np.absolute(u_vec.reshape(n_g_xi,n_g_eta) - f_vec.reshape(n_g_xi,n_g_eta))))
    def testDVec(self):
        tx = np.linspace(0,1,100)
        ty = np.linspace(0,1,100)

        B_dj_tx = self.basis_dj_xi.evaluate(tx)
        B_dj_ty = self.basis_dj_eta.evaluate(ty)

        R_dj = np.kron(B_dj_tx,B_dj_ty)#evaluate_2d_basis(basis_dj_xi,basis_dj_eta,tx,ty)

        d = self.getDVec()

        detJ = np.matmul(np.transpose(d),np.transpose(R_dj))
        Tx,Ty = get_2d_grid(tx,ty)

        #Compute detJ
        detJ2 = self.getDeterminantJ(tx,ty)

        print(np.max(np.abs(detJ-detJ2)))
    def testLiaoHessian(self):
        if not hasattr(self,'M_dj'):
            print('Generate Mass Matrix for computation of d vector')
            self.generateMassMatrix()

        fx = np.zeros(n)
        fx[0] = 1
        fx[n-1] = 1
        fy = np.zeros(n)
        fy[0] = 1
        fy[n-1] = 1

        Fx,Fy = get_2d_grid(fx,fy)
        Fixed_x = np.argwhere(Fx == 1)
        Fixed_y = np.argwhere(Fy == 1)
        Fixed = np.union1d(Fixed_x,Fixed_y)

        P.fix(Fixed,Fixed)

        # JacDVec = P.getJacDVecFixed()
        #
        nc = n*n
        nf = len(Fixed)
        nnf = nc-nf
        #
        Cxnf,Cynf = P.getNonFixedCxy()
        x = np.zeros(nnf*2)
        x[0:nnf] = Cxnf
        x[nnf:nnf*2] = Cynf

        tmpx = x

        l,dl,H = P.getLiaoAndHessian(tmpx)

        # Test of hessian

        np.random.seed(70)
        vx = np.random.rand(nnf)
        vy = np.random.rand(nnf)
        v = np.concatenate((vx,vy),axis = 0)

        Eps = 1/(2**np.arange(0.0,15.0))
        err0 = Eps*0
        err1 = Eps*0
        err2 = Eps*0
        iter = 0

        for eps in Eps:
            guess0 = l
            guess1 = l + eps*dl.dot(v)
            guess2 = l + eps*dl.dot(v) + eps**2/2 * v.dot(H.dot(v))

            exact = P.getLiao(tmpx + eps*v)
            print(guess1-exact, ' , ', guess2-exact)

            err0[iter] = np.abs(guess0-exact)
            err1[iter] = np.abs(guess1-exact)
            err2[iter] = np.abs(guess2-exact)
            iter += 1

        print()
        plt.figure(2)
        plt.loglog(Eps,err0,'b')
        plt.loglog(Eps,err1,'r')
        plt.loglog(Eps,err2,'g')

        plt.loglog(Eps,15000*Eps,'b--')
        plt.loglog(Eps,15000*np.power(Eps,2),'r--')
        plt.loglog(Eps,250*np.power(Eps,3),'g--')
        plt.show()
        print('End of test')
    def testJigsaw(self):
        p = 2;
        knots = generate_knot_vec(np.array([1,2,3,4,5,6,7])/8.0,p)
        n = len(knots) - p - 1

        #p1x = np.array([0,4,4,2,2,8,8,6,6,10])
        #p1y = np.array([0,0,2,2,4,4,2,2,0,0])
        p1x=np.array([0,3,4,3,4,6,7,6,7,10]).T; # Even less curly
        p1y=np.array([0,0,1,1.5,3,3,1.5,1,0,0]).T; # Even less curly
        p2x = p1x
        p2y = 10-p1y

        q1x = -p1y
        q1y = p1x
        q2x = 10 - q1x
        q2y = q1y

        Cx,Cy = SpringModel(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y)
        Cx = Cx.ravel()
        Cy = Cy.ravel()

        # Cx,Cy = GordonHall(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y)
        # Cx = Cx.reshape(100)
        # Cy = Cy.reshape(100)
        # plt.figure()
        # plt.plot(p1x,p1y,'o')
        # plt.plot(p2x,p2y,'o')
        # plt.plot(q1x,q1y,'o')
        # plt.plot(q2x,q2y,'o')
        # plt.plot(Cx,Cy,'k.')
        # plt.xlim([-2,12])
        # plt.ylim([-2,12])
        # plt.show()

        # cx = get_greville()#np.linspace(0,1,n) + np.random.rand(n)*0.1
        # cy = #np.linspace(0,1,n) + np.random.rand(n)*0.1
        P = Parametrization(p,p,knots,knots,Cx,Cy)
        # P.testLiaoHessian()

        P.loadCxy(fname = 'Jigsaw2')
        P.loadz('Jigsaw2_z',delta = 0.01)

        P.plot()

        start = timer()
        P.getParametrization(p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y,method = 'Liao')
        end = timer()
        print('timespend : ',end-start)
        P.plot()
        P.saveCxy(fname = 'Jigsaw2_Liao2')
        plt.show()

    ## Some stuff for Helmholtz
    def getInitialPara_Helmholtz():
        pass

generateParametrizationOfDisc(2,2)
