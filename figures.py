#!/usr/bin/env python
from __future__ import division
import numpy as np
from scipy.linalg import block_diag, solve_triangular
from matplotlib import pyplot as plt
import pydare

##########
#  util  #
##########

def lambda_max(G):
    return sorted(np.linalg.eigvals(G), key=np.abs)[-1]

def rho(G):
    return np.abs(lambda_max(G))

def sigma_max(A):
    return np.linalg.svd(A,compute_uv=0).max()

##############################
#  random matrix generation  #
##############################

def rand_psd(n,rank=None):
    if rank is None:
        rank = np.random.randint(1,n+1)
    A = np.random.normal(size=(n,rank))
    return A.dot(A.T)

#########################
#  standard splittings  #
#########################

def split_gaussseidel(J):
    'J = L-U with U strictly upper-triangular'
    return np.tril(J), -np.triu(J,1)

def split_jacobi(J):
    'J = D-A'
    A = J.copy()
    A.flat[::A.shape[0]+1] = 0
    return np.diag(np.diag(J)), -A

def update(M,N):
    return np.linalg.solve(M,N)

########################
#  Hogwild splittings  #
########################

def even_partition(n,k):
    return np.array_split(np.arange(n),k)

def split_blockdiag(J,partition):
    blockoffdiag = J.copy()
    blocks = []
    for indices in partition:
        blocks.append(blockoffdiag[indices[:,None],indices])
        blockoffdiag[indices[:,None],indices] = 0
    return blocks, -blockoffdiag

def split_hogwild(J,partition):
    blocks, A = split_blockdiag(J,partition)
    Bs,Cs = zip(*(split_gaussseidel(diagblock) for diagblock in blocks))
    return A,Bs,Cs

def update_matrix_hogwild(J,partition,q):
    n = J.shape[0]
    A,Bs,Cs = split_hogwild(J,partition)

    BinvCs = [np.linalg.solve(B,C) for B,C in zip(Bs,Cs)]
    BinvCqs = [np.linalg.matrix_power(BinvC,q) for BinvC in BinvCs]

    BinvC = block_diag(*BinvCs)
    BinvCq = block_diag(*BinvCqs)
    BinvA = np.vstack([np.linalg.solve(B,A[indices,:]) for B,indices in zip(Bs,partition)])

    # TODO write this with (B-C)^{-1} A
    return BinvCq + (np.eye(n) - BinvCq).dot(np.linalg.solve(np.eye(n) - BinvC, BinvA))

#################################
#  spectral radius computation  #
#################################

def gaussseidel_radius(J):
    M,N = split_gaussseidel(J)
    return rho(solve_triangular(M,N,lower=True,overwrite_b=True))

def Tblock_radius(J,partition):
    blocks, A = split_blockdiag(J,partition)
    return max(gaussseidel_radius(b) for b in blocks)

def Tind_radius(J,partition):
    blocks, _ = split_blockdiag(J,partition)
    blocks = block_diag(*blocks)
    return rho(np.linalg.solve(*split_gaussseidel(blocks))) # TODO can be more efficient

############################
#  covariance computation  #
############################

def process_cov(update, injected_cov):
    return pydare.dlyap(update, injected_cov)

def splitting_cov(M,N,raw_injected_cov=None):
    if raw_injected_cov is None: raw_injected_cov = M.T+N
    return process_cov(np.linalg.solve(M,N),
                       np.linalg.solve(M,np.linalg.solve(M,raw_injected_cov).T).T)

def hog_cov(J,partition,q):
    _blocks, A = split_blockdiag(J,partition)
    blockD = block_diag(*_blocks)

    B,C = split_gaussseidel(blockD)
    sub_inf_gibbs = splitting_cov(B,C)
    BinvCq = np.linalg.matrix_power(np.linalg.solve(B,C),q)
    sub_gibbs = sub_inf_gibbs - BinvCq.dot(sub_inf_gibbs).dot(BinvCq.T) # Dtilde

    T = BinvCq + (np.eye(B.shape[0]) - BinvCq).dot(np.linalg.solve(B-C,A))

    return process_cov(T, sub_gibbs)

def cov_errors_onblockdiagonal(sigma1,sigma2,P):
    blocks1, _ = split_blockdiag(sigma1,P)
    blocks2, _ = split_blockdiag(sigma2,P)
    return np.linalg.norm([np.linalg.norm(b1-b2) for b1, b2 in zip(blocks1,blocks2)])

def cov_errors_offblockdiagonal(sigma1,sigma2,P):
    _, OBD1 = split_blockdiag(sigma1,P)
    _, OBD2 = split_blockdiag(sigma2,P)
    return np.linalg.norm(OBD1 - OBD2)

#######################
#  figure generation  #
#######################

def fig1b():
    n = 24
    P = even_partition(n,4)

    import matplotlib
    matplotlib.rcParams.update({'font.size': 20})

    pairs = []
    for i in range(500):
        J = rand_psd(n,rank=n) + np.random.uniform(low=0.5*n,high=n)*np.eye(n)
        pairs.append(( rho(update_matrix_hogwild(J,P,1)), rho(update_matrix_hogwild(J,P,1024)) ))
    pairs = np.array(pairs)

    plt.figure(figsize=(8,5))
    plt.plot(pairs[:,0], pairs[:,1], 'bx')
    plt.xlim(0.75,1.25)
    plt.ylim(0.75,1.25)
    plt.vlines(1,0,2,color='r',linestyles='dashed')
    plt.plot([0,2],[1,1],'r--')
    plt.xlabel(r'$\rho(T)$, $q=1$',fontsize=24)
    plt.ylabel(r'$\rho(T)$, $q=\infty$',fontsize=24)
    plt.gcf().subplots_adjust(bottom=0.15)

def fig1cd():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})

    partition_elt_size=50
    num_partitions=3
    q = 2

    n = partition_elt_size*num_partitions
    P = even_partition(n,num_partitions)

    J = rand_psd(n,rank=n)
    blocks, blockOffD = split_blockdiag(J,P)
    blockD = block_diag(*blocks)

    ts = np.linspace(0,0.2,25)

    ### c

    plt.figure(figsize=(8,5))

    trunc_cov = block_diag(*(np.linalg.inv(b) for b in blocks))
    true_covs = [np.linalg.inv(blockD - t*blockOffD) for t in ts]

    radius = Tind_radius(J,P)

    plt.plot(ts,[cov_errors_onblockdiagonal(trunc_cov,true_cov,P)
        for true_cov in true_covs],'kx-',label=r'$A=0$')

    allerrs = []
    for q in [1,2,3,4]:
        errs = [cov_errors_onblockdiagonal(true_cov, hog_cov(blockD - t*blockOffD,P,q),P)
                for t,true_cov in zip(ts,true_covs)]
        allerrs.append(errs)
        plt.plot(ts,errs,label=(r'$\rho(B^{-1}C)^q=%0.3f$' % radius**q))

    plt.ylim(0,max(max(e) for e in allerrs))
    plt.legend(loc='best',fontsize=18).get_frame().set_facecolor('1.0')
    plt.xlabel(r'$t$',fontsize=18)
    plt.ylabel('block diagonal error')
    plt.gcf().subplots_adjust(left=0.15,bottom=0.11)

    ### d

    plt.figure(figsize=(8,5))

    trunc_cov = block_diag(*(np.linalg.inv(b) for b in blocks))
    true_covs = [np.linalg.inv(blockD - t*blockOffD) for t in ts]

    radius = Tind_radius(J,P)

    plt.plot(ts,[cov_errors_offblockdiagonal(trunc_cov,true_cov,P)
        for true_cov in true_covs],'kx-',label=r'$A=0$')

    allerrs = []
    for q in [1,2,3,4]:
        errs = [cov_errors_offblockdiagonal(true_cov, hog_cov(blockD - t*blockOffD,P,q),P)
                for t,true_cov in zip(ts,true_covs)]
        allerrs.append(errs)
        plt.plot(ts,errs,label=(r'$\rho(B^{-1}C)^q=%0.3f$' % radius**q))

    plt.ylim(0,max(max(e) for e in allerrs))
    plt.legend(loc='best',fontsize=18).get_frame().set_facecolor('1.0')
    plt.xlabel(r'$t$',fontsize=18)
    plt.ylabel('off-block-diagonal error')
    plt.gcf().subplots_adjust(left=0.15,bottom=0.11)

##########
#  main  #
##########

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 1:
        if len(sys.argv) == 2 and sys.argv[1].startswith('--seed='):
            np.random.seed(int(sys.argv[1][7:]))
        else:
            print 'usage: %s [--seed=INT]' % sys.argv[0]
            sys.exit(1)

    fig1b()
    plt.figure(1); plt.savefig('fig1b.pdf',transparent=True)
    fig1cd()
    plt.figure(2); plt.savefig('fig1c.pdf',transparent=True)
    plt.figure(3); plt.savefig('fig1d.pdf',transparent=True)
    plt.show()

