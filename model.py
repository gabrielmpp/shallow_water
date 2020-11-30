import numpy as np

class shallow_water():
    '''
    This class initializes the parameters of the shallow water model
    and the methods for computing the analytical solution for that
    particular set of parameters.
    '''

    def __init__(self, Lx, Ly, f_0 = 1e-4, beta=1e-11, g=10, gamma = 1e-6, rho = 1e3, H = 1e3, tau_0 = 0.2):

        self.Lx = Lx
        self.Ly = Ly
        self.f_0 = f_0
        self.beta = beta
        self.g = g
        self. gamma = gamma
        self.rho = rho
        self.H = H
        self.tau_0 = tau_0
        print("Model initialized")

    def calc_tau(self, y, L):
        tau_x = self.tau_0 * (-np.cos(np.pi * y/L))
        tau_y = 0
        return tau_x, tau_y

    def calc_f1_f2(self, x):

        epsilon = self.gamma / (self.Ly * self.beta)
        b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2))/(2*epsilon)
        a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2))/(2*epsilon)

        f1 = np.pi*(1 + ((np.exp(a)-1)*np.exp(b*x) + (1-np.exp(b))*np.exp(a*x))/(np.exp(b)-np.exp(a)))
        f2 = ((np.exp(a) - 1) * b * np.exp(b * x) + (1 - np.exp(b)) * a * np.exp(a * x))/(np.exp(b)-np.exp(a))
        return f1, f2

    def analytical_solution(self, x, y, eta):
        f1, f2 = self.calc_f1_f2(x= x/self.Lx)
        k1 = self.tau_0/(np.pi*self.gamma*self.rho*self.H)
        k2 = self.f_0*self.Lx/self.g
        k3 = np.pi*y/self.Lx

        ust =  -k1*f1*np.cos(k3)
        vst =   k1*f2*np.sin(k3)

        eta_0 = -0.2
        etast = eta_0 + k1*k2*((self.gamma/(self.f_0*np.pi))*f2*np.cos(k3) +
                                 (1/np.pi)*f1*(np.sin(k3)*(1+self.beta*y/self.f_0)+
                                 self.beta*self.Lx/(self.f_0*np.pi) * np.cos(k3)) )
        return  ust, vst, etast
