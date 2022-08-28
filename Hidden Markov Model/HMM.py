import numpy as np

class HMM:
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N
        self.M = M
        self.pi = pi
        self.A = A
        self.B = B

    def rdistribution(self, dist): 
        r = np.random.rand()
        for ix, p in enumerate(dist):
            if r < p: 
                return ix
            r -= p

    def generate(self, T):
        i = self.rdistribution(self.pi)  
        o = self.rdistribution(self.B[i])  
        observed_data = [o]
        for _ in range(T-1):        
            i = self.rdistribution(self.A[i])
            o = self.rdistribution(self.B[i])
            observed_data.append(o)
        return observed_data

def prob_calc(O):
    alpha = pi * B[:, O[0]]
    for o in O[1:]:
        alpha_next = np.empty(4)
        for j in range(4):
            alpha_next[j] = np.sum(A[:,j] * alpha * B[j,o])
        alpha = alpha_next
    return alpha.sum()

def viterbi_decode(O):
    T, o = len(O), O[0]
    delta = pi * B[:, o]
    varphi = np.zeros((T, 4), dtype=int)
    path = [0] * T
    for i in range(1, T):
        delta = delta.reshape(-1, 1)     
        tmp = delta * A
        varphi[i, :] = np.argmax(tmp, axis=0)
        delta = np.max(tmp, axis=0) * B[:, O[i]]
    
    path[-1] = np.argmax(delta)
    for i in range(T-1, 0, -1):
        path[i-1] = varphi[i, path[i]]
    return path

if __name__=="__main__":
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    A = np.array([
        [0,  1,  0, 0],
        [0.4, 0, 0.6, 0],
        [0, 0.4, 0, 0.6],
    [0, 0, 0.5, 0.5]])
    
    B = np.array([
        [0.5, 0.5],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.3, 0.7]])
    
    N = 4
    M = 2

    hmm = HMM(N, M, pi, A, B)
    O = hmm.generate(5)
    print(O)
    print(prob_calc(O))
    print(viterbi_decode(O))