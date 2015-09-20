#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <gsl/gsl_rng.h>
#include "secorder_rec_1p.hpp"
#include "calc_stats_1p.hpp"


using namespace std;

int main(int argc, char *argv[]) {
  

  ///////////////////////////////////////////////////////////////
  // Read seven input parameters 
  // N_nodes      number of nodes in the network
  // p            the probability of any one given connection
  // alpha_recip  determines covariance of reciprocal connections
  // alpha_conv   determines covariance of convergent connections
  // alpha_div    determines covariance of divergent connections
  // alpha_chain  determines covariance of chain connections
  // seed         seed for the random number generator (optional)
  ///////////////////////////////////////////////////////////////

 
  if(argc < 7 || argc > 8) {
    cerr << "Requires six or seven parameters: N_nodes p alpha_recip alpha_conv alpha_div alpha_chain [seed]\n";
    exit(-1);
  }

  int N_nodes = atoi(argv[1]);
  double p = atof(argv[2]);
  double alpha_recip = atof(argv[3]);
  double alpha_conv = atof(argv[4]);
  double alpha_div = atof(argv[5]);
  double alpha_chain = atof(argv[6]);

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);


  int deterministic_seed=0;
  int rng_seed = 0;
  if(argc >7) {
    deterministic_seed=1;
    rng_seed = atoi(argv[7]);    
  }

  // set the seed
  // if deterministic_seed, use rng_seed for seed
  if(deterministic_seed)
    gsl_rng_set(rng,rng_seed); 
  // else use time in seconds for the seed
  else
    gsl_rng_set(rng, time(NULL));
  
  gsl_matrix *W;
  W=secorder_rec_1p(N_nodes,p,  alpha_recip, alpha_conv, alpha_div, alpha_chain,
		    rng);
  
    // if failed to generate a matrix, write error and quit
  if(!W) {
    cerr << "Failed to generate the matrix\n";
    return -1;
  }

  // matrix files will be stored in data directory
  // with filename determined by the six or seven input parameters
  mkdir("data",0755);  // make data directory, if it doesn't exist
  char FNbase[200];
  if(deterministic_seed)
    sprintf(FNbase,"_%i_%1.3f_%1.3f_%1.3f_%1.3f_%1.3f_%i",N_nodes,p,alpha_recip, alpha_conv, alpha_div, alpha_chain, rng_seed);
  else
    sprintf(FNbase,"_%i_%1.3f_%1.3f_%1.3f_%1.3f_%1.3f",N_nodes,p,alpha_recip, alpha_conv, alpha_div, alpha_chain);
  
  char FN[200];
  FILE *fhnd;
  strcpy(FN, "data/w");
  strcat(FN, FNbase);
  strcat(FN, ".dat");
  fhnd = fopen(FN, "w");
  if(fhnd==NULL) {
    cerr << "Couldn't open outfile file " << FN << "\n";
    exit(-1);
  }

  for(int i=0; i<N_nodes;i++) {
    for(int j=0; j<N_nodes; j++) {
      fprintf(fhnd, "%i ", gsl_matrix_get(W,i,j)>1.0);
    }
    fprintf(fhnd,"\n");
  }
  fclose(fhnd);

  

  ////////////////////////////////////////////////////////////
  // Calculate the covariance structure the adjacency matrix
  // This should approximately agree with the input alphas
  ////////////////////////////////////////////////////////////

  cout << "Testing statistics of W ...";
  cout.flush();
  double phat, alphahat_recip, alphahat_conv, alphahat_div,
    alphahat_chain, alphahat_other;
  
  calc_phat_alphahat_1p(W, N_nodes, phat, alphahat_recip, 
			alphahat_conv, alphahat_div,
			alphahat_chain, alphahat_other);

  strcpy(FN, "data/stats");
  strcat(FN, FNbase);
  strcat(FN, ".dat");
  fhnd = fopen(FN, "w");
  if(fhnd==NULL) {
    cerr << "Couldn't open outfile file " << FN << "\n";
    exit(-1);
  }
  fprintf(fhnd, "%e %e %e %e %e %e\n", phat, alphahat_recip, 
	  alphahat_conv, alphahat_div, alphahat_chain, alphahat_other);
  fclose(fhnd);


  cout << "done\n";
  cout << "\nActual statistics of matrix:\n";
  cout << "phat = " << phat << "\n";
  cout << "alphahats:\n";
  cout << "alpha_recip = " << alphahat_recip
       << ", alpha_conv = " << alphahat_conv
       << ", alpha_div = " << alphahat_div
       << ", alpha_chain = " << alphahat_chain
       << ", alpha_other = " << alphahat_other << "\n";
  
}
