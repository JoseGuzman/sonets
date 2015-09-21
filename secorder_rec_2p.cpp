#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>

#include "calc_sqrtcov_rec_2p.hpp"
#include "calc_rhos.hpp"
#include "calc_stats_2p.hpp"
#include "secorder_rec_2p.hpp"

using namespace std;


// declare auxiliary function
int gen_corr_gaussian(const int N_nodes[2], 
		      double (*sqrt_diag)[2], double (*sqrt_recip)[2],
		      double (*sqrt_conv)[2][2], double (*sqrt_div)[2][2], 
		      double (*sqrt_chain)[2][2],
		      gsl_matrix *thevars, gsl_rng *r);


//////////////////////////////////////////////////////////////////////
// secorder_rec_1p
//
// Generate Bernoulli random matrix corresponding to
// a second order network of two population, indexed by 0 and 1,
// containing N_nodes[0] and N_nodes[1] nodes.
//
// The target statistics of connectivity are given by arguments
//
// First order statistic
// p[i][j]: probability of a connection from one node in population j
//          onto a second node in population i
//
// Second order statistics
// alpha_recip[i][j]: reciprocal connection parameter for connections
//       between populations i and j (i<j)
//       alpha_recip must be > -1  (not only restriction)
// alpha_conv_hom[i][j]: convergent connection parameter for connections
//       from a pair of nodes in population j onto a third node
//       in population i
//       alpha_conv_hom must be positive (not only restriction)
// cc_conv_mixed[i][j][k]: convergent connection parameter for connections
//       from a pair of nodes in distinct population j < k
//       onto a third node in population i
//       For two populations, only possibilities are [0][0][1] and [1][0][1]
//       cc_conv_mixed must be between -1 and 1 (not only restriction)
// alpha_div_hom[i][j]: divergent connection parameter for connections
//       from a node in population j onto a pair of nodes in population i
//       alpha_div_hom must be positive (not only restriction)
// cc_div_mixed[i][j][k]: convergent connection parameter for connections
//       from a node in population k onto a pair of nodes in distinct
//       population i < j.
//       For two populations, only possibilities are [0][1][0] and [0][1][1]
//       cc_div_mixed must be between -1 and 1 (not only restriction)
// cc_chain[i][j][k]: chain connection parameter for connections from
//       a neuron in population k onto a second neuron in population j
//       and from that second neuron in population k onto a third
//       neuron in population i
//       cc_chain must be between -1 and 1 (not only restriction)
//
// argument rng is pointer to an initialized gsl random number generator
//
// returns:
// 0 if unsuccessful at generating matrix
//   (not all combinations of alpha are valid)
// a pointer to allocated gsl_matrix of size N_nodes[0]+N_nodes[1]
// by N_nodes[0]+N_nodes[1] if successful at generating matrix
// 
// notation convenction is that entry (i,j) is
// connection from node j onto node i
// the first N_nodes[0] indices refer to population 0 and 
// the second N_nodes[1] indices refer to population 1
////////////////////////////////////////////////////////////////////////

gsl_matrix* secorder_rec_2p(int N_nodes[2], double (*p)[2],
			    double (*alpha_recip)[2], 
			    double (*alpha_conv_hom)[2], 
			    double (*cc_conv_mixed)[2][2], 
			    double (*alpha_div_hom)[2],
			    double (*cc_div_mixed)[2][2],
			    double (*cc_chain)[2][2], gsl_rng *r) {

  int calc_covs=0;  // if nonzero, calculate covariance of Gaussian
  
  int print_palpha = 1; // print out target values of p and alpha
  int print_rho = 1;    // print out values of rho
  int print_sqrt = 0;   // print out values of square root of covariance

  int status;
  int N_nodes_tot= N_nodes[0]+N_nodes[1];

  
  cout << "Beginning secorder_rec_2p with N_nodes = " << N_nodes[0]
       << ", " << N_nodes[1]
       << "\n";

  if(print_palpha) {
    cout << "p = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	cout << p[i][j] << " ";
    cout << "\n";
    cout << "alpha_recip = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	cout << alpha_recip[i][j] << " ";
    cout << "\n";
    cout << "alpha_conv_hom = "; 
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	  cout << alpha_conv_hom[i][j] << " ";
    cout << "\n";
    cout << "cc_conv_mixed = "; 
    for(int i=0; i<2; i++)
      for(int j=0; j<1; j++)
	for(int k=j+1; k<2; k++)
	  cout << cc_conv_mixed[i][j][k] << " ";
    cout << "\n";
    cout << "alpha_div_hom = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	  cout << alpha_div_hom[i][j] << " ";
    cout << "\n";
    cout << "cc_div_mixed = ";
    for(int i=0; i<1; i++)
      for(int j=i+1; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << cc_div_mixed[i][j][k] << " ";
    cout << "\n";
    cout << "cc_chain = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << cc_chain[i][j][k] << " ";
    cout << "\n";
    cout.flush();
  }

  //////////////////////////////////////////////////////////////
  // Step 1: Transform desired alphas for the Bernoulli matrix
  // into the required covariance structure (rhos) 
  // of the underlying Gaussian matrix
  // The rhos implicitly define the Gaussian's covariance matrix
  //////////////////////////////////////////////////////////////

  // rho_recip[i][j] is covariance between reciprocal connections
  // between a node in population i and a node in population j, i <= j
  double rho_recip[2][2];

  // rho_conv[i][j][k] is covariance between convergent connections
  // from a pair of nodes in populations j and k, j<=k
  // onto a third node in population i
  double rho_conv[2][2][2];

  // rho_div[i][j][k] is covariance between divergent connections
  // from a node in population k onto a pair of nodes in populations
  // i and j, i<=j
  double rho_div[2][2][2];

  // rho_chain[i][j][k] is covariance between chain connections
  // from a node in population k onto a second node in population j
  // and from the same node in population j onto a third node in population i
  double rho_chain[2][2][2];

  for(int i=0; i<2; i++) {
    // for reciprocal connections, calculate required covariance
    // in Gaussian so that Bernoulli random variables will have
    // the covariance determined by alpha_recip
    for(int j=i; j<2; j++) {
      rho_recip[i][j] = calc_rho_given_alpha(p[i][j], p[j][i], 
					     alpha_recip[i][j],status);
      if(status)
	return 0;
    }
    
    // for convergent connections from a pair of neurons in the same
    // population onto a third neuron in any population,
    // calculate required covariance
    // in Gaussian so that Bernoulli random variables will have
    // the covariance determined by alpha_conv_hom
    for(int j=0; j<2; j++) {
      rho_conv[i][j][j]
	= calc_rho_given_alpha(p[i][j], p[i][j], 
			       alpha_conv_hom[i][j],status);
      if(status)
	return 0;
    }

    // for divergent connections onto a pair of neurons in the same
    // population from a third neuron in any population,
    // calculate required covariance
    // in Gaussian so that Bernoulli random variables will have
    // the covariance determined by alpha_div_hom
    for(int j=0; j<2; j++) {
      rho_div[i][i][j]
	= calc_rho_given_alpha(p[i][j], p[i][j], 
			       alpha_div_hom[i][j],status);
      if(status)
	return 0;
    }
  }
  
  for(int i=0; i<2; i++) {
    // for convergent connections from two nodes in different population
    // onto a third node in any population, calculate covariance
    // of Gaussian from the homogeneous convergent Gaussian covariances
    // and cc_conv_mixed
    for(int j=0; j<1; j++) 
      for(int k=j+1; k<2; k++) {
	double temp = rho_conv[i][j][j]*rho_conv[i][k][k];
	rho_conv[i][j][k] = sqrt(temp)*cc_conv_mixed[i][j][k];

      }

    // for divergent connections onto two nodes in different population
    // from a third node in any populations, calculate covariance
    // of Gaussian from the homogeneous divergent Gaussian covariances
    // and cc_div_mixed
    for(int j=i+1; j<2; j++) 
      for(int k=0; k<2; k++) {
	double temp = rho_div[i][i][k]*rho_div[j][j][k];
	rho_div[i][j][k] = sqrt(temp)*cc_div_mixed[i][j][k];
      }

    // for chain connections from neuron one to neuron two to neuron three,
    // calculate covariance of Gaussian from cc_chain,  the homogeneous 
    // convergent Gaussian covariance from neuron one's population
    // onto neuron two's population, and the homogeneous divergent
    // Gaussian covariance from neuron two's population onto
    // neuron three's population
    for(int j=0; j<2; j++) 
      for(int k=0; k<2; k++) {
	double temp = rho_conv[j][k][k]*rho_div[i][i][j];
	rho_chain[i][j][k] = sqrt(temp)*cc_chain[i][j][k];
      }
  }
  
  if(print_rho) {
    cout << "rho_recip = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	cout << rho_recip[i][j] << " ";
    cout << "\n";
    cout << "rho_conv = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=j; k<2; k++)
	  cout << rho_conv[i][j][k] << " ";
    cout << "\n";
    cout << "rho_div = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << rho_div[i][j][k] << " ";
    cout << "\n";
    cout << "rho_chain = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << rho_chain[i][j][k] << " ";
    cout << "\n";
    cout.flush();
  }

  ////////////////////////////////////////////////////////////
  // Step 2: Take the square root of the Gaussian's covariance matrix
  // 
  //
  // This step will not always succeed because some combinations of
  // rhos do not lead to a valid covariance matrix.
  // 
  // calc_sqrtcov_given_rhos only accepts combinations 
  // of rhos that are valid in the limit of large networks
  ////////////////////////////////////////////////////////////
  
  double sqrt_diag[2][2];
  double sqrt_recip[2][2];
  double sqrt_conv[2][2][2];
  double sqrt_div[2][2][2];
  double sqrt_chain[2][2][2];

  status = calc_sqrtcov_given_rhos
    (N_nodes, p, rho_recip, rho_conv, rho_div, rho_chain,
     sqrt_diag, sqrt_recip, sqrt_conv, sqrt_div, sqrt_chain);

  if(status) {
    cerr << "Could not find real square root\n";
    return 0;
  }

  if(print_sqrt) {
    cout << "sqrt_diag = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	cout << sqrt_diag[i][j] << " ";
    cout << "\n";
    cout << "sqrt_recip = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	cout << sqrt_recip[i][j] << " ";
    cout << "\n";
    cout << "sqrt_conv = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=j; k<2; k++)
	  cout << sqrt_conv[i][j][k] << " ";
    cout << "\n";
    cout << "sqrt_div = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << sqrt_div[i][j][k] << " ";
    cout << "\n";
    cout << "sqrt_chain = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << sqrt_chain[i][j][k] << " ";
    cout << "\n";
    cout.flush();
  }

  ////////////////////////////////////////////////////////////
  // Step 3: Use the square root of the covariance matrix
  // to generate the Gaussian matrix with the desired 
  // covariance structure.
  // Simply need to generate a vector of independent Gaussians
  // and multiply by the covariance matrix
  ////////////////////////////////////////////////////////////

  
  gsl_matrix *W_gaus = gsl_matrix_alloc(N_nodes_tot, N_nodes_tot);

  cout << "Generating Gaussian matrix...";
  cout.flush();
  // calculate Gaussian matrix
  gen_corr_gaussian(N_nodes, sqrt_diag, sqrt_recip, sqrt_conv, 
		    sqrt_div, sqrt_chain, W_gaus, r);
  cout << "done\n";
  cout.flush();


  ////////////////////////////////////////////////////////////
  // Optional step 4: Calculate the covariance structure
  // of the Gaussian matrix 
  // Then, one can check program output to see if the
  // Gaussian matrix was generated correctly
  ////////////////////////////////////////////////////////////


  if(calc_covs) {

    cout << "Calculating correlations...";
    cout.flush();
    double sigma[2][2];
    double cov_recip[2][2];
    double cov_conv[2][2][2];
    double cov_div[2][2][2];
    double cov_chain[2][2][2];
    double cov_noshare=0.0;

    calc_gaus_covs(W_gaus, N_nodes,
		   sigma,cov_recip,
		   cov_conv, cov_div,
		   cov_chain,cov_noshare);

    cout << "done\n";

    cout << "sigma = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	cout << sigma[i][j] << " ";
    cout << "\n";
    cout << "cov_recip = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	cout << cov_recip[i][j] << " ";
    cout << "\n";
    cout << "cov_conv = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=j; k<2; k++)
	  cout << cov_conv[i][j][k] << " ";
    cout << "\n";
    cout << "cov_div = ";
    for(int i=0; i<2; i++)
      for(int j=i; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << cov_div[i][j][k] << " ";
    cout << "\n";
    cout << "cov_chain = ";
    for(int i=0; i<2; i++)
      for(int j=0; j<2; j++)
	for(int k=0; k<2; k++)
	  cout << cov_chain[i][j][k] << " ";
    cout << "\n";
    cout << "cov_noshare = " << cov_noshare << "\n";

    cout.flush();
  }

  ////////////////////////////////////////////////////////////
  // Step 5: Calculate Bernoulli matrix
  // Simply make the Bernoulli variable be 1 
  // if the Gaussian variable is greater than 1
  ////////////////////////////////////////////////////////////

  cout << "Generating Bernoulli matrix...";
  cout.flush();
  // calculate bernoulli matrix
  gsl_matrix *W_ber = gsl_matrix_alloc(N_nodes_tot, N_nodes_tot);

  for(int i=0; i<N_nodes_tot;i++) {
    for(int j=0; j<N_nodes_tot; j++) {
      gsl_matrix_set(W_ber,i,j,gsl_matrix_get(W_gaus,i,j)>1.0);
    }
  }

  // free Gaussian matrix
  gsl_matrix_free(W_gaus);


  cout << "done\n";
  
  // return Bernoulli matrix
  return W_ber;


}


///////////////////////////////////////////////////////
// gen_corr_gaussian
// Generate correlated Gaussian given the square of the 
// covariance matrix determined by sqrtcov_pars
///////////////////////////////////////////////////////

int gen_corr_gaussian(const int N_nodes[2], 
		      double (*sqrt_diag)[2], double (*sqrt_recip)[2],
		      double (*sqrt_conv)[2][2], double (*sqrt_div)[2][2], 
		      double (*sqrt_chain)[2][2],
		      gsl_matrix *thevars, gsl_rng *r) {

  gsl_matrix_set_zero(thevars);
  size_t tda= thevars->tda;
  double *thedata = thevars->data;
    
  int N_shift[2];
  N_shift[0]=0;
  N_shift[1]=N_nodes[0];

  int N_nodes_tot = N_nodes[0]+N_nodes[1];
  // index row and column sums by node number
  double row_sums[N_nodes_tot];
  double col_sums[N_nodes_tot];

  // generate N_nodes_tot*(N_nodes_tot-1) independent Gaussians
  // then multipy by square root of covariance matrix 
  // determined by sqrtcov
  
  for(int ntype1=0; ntype1<2; ntype1++)
    for(int ntype2=0; ntype2<2; ntype2++)  {
      for(int i=0; i<N_nodes_tot; i++)
	row_sums[i]=col_sums[i]=0.0;

      for(int i=N_shift[ntype1]; i<N_nodes[ntype1]+N_shift[ntype1]; i++) {
	size_t i_tda=i*tda;
      
	for(int j=N_shift[ntype2]; j<N_nodes[ntype2]+N_shift[ntype2]; j++) {
	  // no connection from node onto itself
	  if(i==j)
	    continue;

	  double gaus_ind= gsl_ran_gaussian(r,1.0);

	  // add diagonal contribution
	  thedata[i_tda + j] += gaus_ind*(sqrt_diag[ntype1][ntype2]
					 -sqrt_conv[ntype1][ntype2][ntype2]
					 -sqrt_div[ntype1][ntype1][ntype2]);
	  
	  // add reciprocal contribution
	  thedata[j*tda + i] += gaus_ind*(sqrt_recip[ntype1][ntype2]
					  -sqrt_chain[ntype1][ntype2][ntype1]
					  -sqrt_chain[ntype2][ntype1][ntype2]);
	  
	  // row and columns sums for conv/div/chain contributions
	  row_sums[i] += gaus_ind;
	  col_sums[j] += gaus_ind;
	}
      }
      for(int ntype3=0; ntype3<2; ntype3++) {
	for(int k=N_shift[ntype3]; k<N_nodes[ntype3]+N_shift[ntype3]; k++) {
	  size_t k_tda=k*tda;
	
	  for(int i=N_shift[ntype1]; i<N_nodes[ntype1]+N_shift[ntype1]; i++) {
	    int i_tda=i*tda;
	    // no connection from node onto itself
	    if(i==k)
	      continue;
      
	    thedata[i_tda+k] += row_sums[i]*sqrt_conv[ntype1][ntype3][ntype2];
	    thedata[k_tda+i] += row_sums[i]*sqrt_chain[ntype3][ntype1][ntype2];
	  }

	  for(int j=N_shift[ntype2]; j<N_nodes[ntype2]+N_shift[ntype2]; j++) {
	    // no connection from node onto itself
	    if(j==k)
	      continue;
	    int j_tda=j*tda;
	  
	    thedata[k_tda+j] += col_sums[j]*sqrt_div[ntype3][ntype1][ntype2];
	    thedata[j_tda+k] += col_sums[j]*sqrt_chain[ntype1][ntype2][ntype3];
	  }
	}
      }
    }

  return 0;
}
