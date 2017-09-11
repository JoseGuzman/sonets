# sonets
Generates Second Order Networks (SONETs) with prescribed second order motif frequencies

## Overview

C++ code that generates random directed unweighted graphs with prescribed second order statistics.
Additional structure (i.e., higher-order statistics) are approximately minimized using a near maximum entropy approach.

Current version of code can generate a single statistically homogeneous network or a two population network
where the statistics can vary as a function of the populations participating in each connection.

Second order statistics are prescribed based on frequencies of four types of two-edge motifs (subnetworks containing two edges):

* reciprocal motifs (two nodes reciprocally connected:  o <-> o)
* convergent motifs (two nodes each connecting onto a third node:  o -> o <- o)
* divergent motifs (two nodes each receving a connection from a third node: o <- o -> o)
* chain motifs (three nodes connected in sequence: o -> o -> o)

Reference: L. Zhao, B. Beverlin II, T Netoff, and D. Q. Nykamp.  Synchronization from second order network connectivity statistics.
*Frontiers in Computational Neuroscience*, 5:28, 2011.  [Publisher's web site](http://dx.doi.org/10.1007/s10827-011-0373-5)

## Requirements

* C++ compiler (such as the [gcc compiler](https://gcc.gnu.org/))
* [GNU Software Library (GSL)](http://www.gnu.org/software/gsl/) 
(If installing a gsl package on a system, such as on Ubuntu, where the development library is separated from the main library,
also install the development library.) 

    sudo apt-get install libgsl-dev

## Usage notes

### Single population network

To generate an adjacency matrix for a single statistically homogeneous network in a C++ program, include the header file secorder_rec_1p.hpp and call secorder_rec_1p as

    gsl_matrix *W = secorder_rec_1p(N_nodes, p, alpha_recip, alpha_conv,
             alpha_div, cc_chain, rng);

where N_nodes is the number of nodes in the network and rng is a pointer to an initialized gsl random number generator.  W is an adjacency matrix where W_ij=1 if there is a connection from node j onto node i and is zero otherwise.

The parameter p is the probability of a single edge, i.e., p = P(W_ij = 1) for any nodes i != j.  The generation algorithm requires that 0 < p <= 0.5.

The alphas specify the probability of a motif W_ij and W_kl according to P(W_ij=1 and W_kl =1) = p^2(1+alpha) where alpha reperesent the appropriate motif parameter, as follows.  If i, j, and k represent distinct nodes, then alpha=alpha_recip for the motifs W_ij and W_ji, alpha=alpha_conv for the motifs are W_ij and W_ik, and alpha=alpha_div for the motifs W_ij and W_kj.  cc_chain specifies the relative magnitude of alpha=alpha_chain for the motifs W_ij and W_jk, as described below.  Since the edge probability is fixed at p, the alphas can be interpreted as specifying the motif frequency or the covariance between edges in the motif.  

Only a small range of parameters alpha_recip, alpha_conv, alpha_div, and cc_chain represent actual networks.  The function secorder_rec_1p returns 0 if it was not able to generate a network.  Given their definition, the range of the alphas might appear that they could range between -1 and 1/p-1.  However, in reality, their range is much smaller.

The parameters alpha_conv and alpha_div must be non-negative, as they determine the variance of the in- and out-degree distributions.  Unless, one of these parameters is the only non-zero alpha, the valid upper limit seems to scale closer to 0.5/p^(2/3).  However, when alpha_conv and alpha_div are close to their maximum values, possible combinations of alpha_recip and alpha_chain are highly restricted.  

The parameter alpha_recip can be negative.  If alpha_recip is close to its maximum value 1/p-1, then network is nearly undirected, which requires alpha_conv to be approximately equal to alpha_div and cc_chain to be approximately 1.  Keeping alpha_recip much smaller allows more flexibility in the other parameters.

The parameter alpha_chain determines the covariance between the in- and out-degrees, the variance of which is determined by alpha_conv and alpha_div, respectively, the limits of alpha_chain depend on alpha_conv and alpha_div.  Hence, to siplify specification of alpha_chain, the function uses the parameters cc_chain, which can range between -1 and 1.  When cc_chain=0, then alpha_chain=0.  When cc_chain=1, alpha_chain is set to its maximum value, which is close to sqrt(alpha_conv*alpha_div).  When cc_chain=-1, alpha_chain is set to its minimum value, which also depends on alpha_conv and alpha_div.

A test program run_secorder can be used to create a standalone program that calls secorder_rec_1p and then tests the statistics of the matrix.  It can be compiled with

    make run_secorder

and then executed with the command

    run_secorder [N_nodes] [p] [alpha_recip] [alpha_conv] [alpha_div] [cc_chain] [optional seed]



### Two population network

For now, see run_secorder_2p.cpp for an example on how to call secorder_rec_2p, which generates a network corresponding to two populations.  In this case, there are four different connection probabilities corresponding to the four different types of connections between the two populations: p_11, p_12, p_21, and p_22.  In addition, there are three types alpha_recips, six different alpha_convs and alpha_divs, and 8 different alpha_chains.  The permitted values of these parameters are strongly interrelated, as in the single population case.
