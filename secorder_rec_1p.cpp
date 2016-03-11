#include <string.h>
#include <time.h>
#include <sys/time.h>


#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>

#include "calc_sqrtcov_rec_1p.hpp"
#include "calc_rhos.hpp"
#include "secorder_rec_1p.hpp"

using namespace std;

extern struct timeval T0;

// declare auxiliary functions
int gen_corr_gaussian(const int N_nodes, double sqrt_diag, double sqrt_recip,
		      double sqrt_conv, double sqrt_div, double sqrt_chain,
              double sqrt_noshare, gsl_matrix_float *thevars, gsl_rng *rng,
              const char *algorithm);
int calc_gaus_covs(gsl_matrix_float *W_gaus, int N_nodes,
		   double &sigma, double &cov_recip,
		   double &cov_conv, double &cov_div,
		   double &cov_chain, double &cov_other);


//////////////////////////////////////////////////////////////////////
// secorder_rec_1p
//
// Generate Bernoulli random matrix corresponding to
// a second order network of one population containing N_nodes nodes
//
// The target statistics of connectivity are given by arguments
//
// First order statistic
// p: probability of a connection
//
// Second order statistics
// alpha_recip: reciprocal connection parameter
// alpha_conv: convergent connection parameter
// alpha_div: divergent connection parameter
// alpha_chain: chain connection parameter
//
// argument rng is pointer to an initialized gsl random number generator
//
// returns:
// 0 if unsuccessful at generating matrix
//   (not all combinations of alpha are valid)
// a pointer to allocated gsl_matrix if successful at generating matrix
// 
// notation convention is that entry (i,j) is
// connection from node j onto node i
////////////////////////////////////////////////////////////////////////

gsl_matrix_float* secorder_rec_1p(int N_nodes, double p,
			    double alpha_recip, double alpha_conv,
			    double alpha_div, double alpha_chain,
			    gsl_rng *rng, const char *algorithm) {

  int calc_covs=0;  // if nonzero, calculate covariance of Gaussian

  int print_palpha = 1; // print out target values of p and alpha
  int print_rho = 1;    // print out values of rho
  int print_sqrt = 0;   // print out values of square root of covariance

  int status;

  struct timeval T1,dT;
  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s N=%i %s (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,N_nodes,algorithm,__FILE__,__LINE__);
  
  if(print_palpha) {
    cout << "p = " << p << "\n";
    cout << "alpha_recip = " << alpha_recip
	 << ", alpha_conv = " << alpha_conv
	 << ", alpha_div = " << alpha_div
	 << ", alpha_chain = " << alpha_chain
	 << "\n";
    cout.flush();
  }

  //////////////////////////////////////////////////////////////
  // Step 1: Transform desired alphas for the Bernoulli matrix
  // into the required covariance structure (rhos) 
  // of the underlying Gaussian matrix
  // The rhos implicitly define the Gaussian's covariance matrix
  //////////////////////////////////////////////////////////////

  double rho_recip, rho_conv, rho_div, rho_chain, rho_noshare;

  // edges that do not share a node are to be uncorrelated
  rho_noshare = 0.0;



  rho_recip = calc_rho_given_alpha(p, p, alpha_recip, status);
  if(status)
    return 0;
  rho_conv = calc_rho_given_alpha(p, p, alpha_conv, status);
  if(status)
    return 0;
  rho_div = calc_rho_given_alpha(p, p, alpha_div, status);
  if(status)
    return 0;

  // if alpha_chain == -3, then let rho_chain be min possible,
  // i.e., - geometric mean of rho_conv and rho_div
  if(alpha_chain <= -3)
    rho_chain = -sqrt(rho_conv*rho_div);
  // if alpha_chain == -2, then let rho_chain be max possible,
  // i.e., geometric mean of rho_conv and rho_div
  else if(alpha_chain <=-2)
    rho_chain = sqrt(rho_conv*rho_div);
  else 
    rho_chain = calc_rho_given_alpha(p, p, alpha_chain, status);
  if(status)
    return 0;

  
  if(print_rho) {
    cout << "rho_recip = " << rho_recip
	 << ", rho_conv = " << rho_conv
	 << ", rho_div = " << rho_div
	 << ", rho_chain = " << rho_chain
	 << ", rho_noshare = " << rho_noshare
	 << "\n";
    cout.flush();
  }
  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t %s step1 (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  
  ///////////////////////////////////////////////////////////////////
  // Step 2: Take the square root of the Gaussian's covariance matrix
  //
  // This step will not always succeed because some combinations of
  // rhos do not lead to a valid covariance matrix.
  // 
  // By default, calc_sqrtcov_given_rhos only accepts combinations 
  // of rhos that are valid in the limit of large networks
  ///////////////////////////////////////////////////////////////////

  double sqrt_diag, sqrt_recip, sqrt_conv, sqrt_div, sqrt_chain, sqrt_noshare;

  status = calc_sqrtcov_given_rhos
    (N_nodes, p, rho_recip, rho_conv, rho_div, rho_chain, rho_noshare,
     sqrt_diag, sqrt_recip, sqrt_conv, sqrt_div, sqrt_chain, sqrt_noshare);

  if(status) {
    cerr << "Could not find real square root\n";
    return 0;
  }
  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t %s step2 (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
 
  if(print_sqrt) {
    cout << "sqrt_diag = " << sqrt_diag
	 << ", sqrt_recip = " << sqrt_recip
	 << ", sqrt_conv = " << sqrt_conv
	 << ", sqrt_div = " << sqrt_div
	 << ", sqrt_chain = " << sqrt_chain
	 << ", sqrt_noshare = " << sqrt_noshare
	 << "\n";
    cout.flush();
  }

  ////////////////////////////////////////////////////////////
  // Step 3: Use the square root of the covariance matrix
  // to generate the Gaussian matrix with the desired 
  // covariance structure.
  // Simply need to generate a vector of independent Gaussians
  // and multiply by the covariance matrix
  ////////////////////////////////////////////////////////////


  gsl_matrix_float *W_gaus = gsl_matrix_float_alloc(N_nodes, N_nodes);

  cout << "Generating gaussian matrix...";
  cout.flush();

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t %s step3 (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  fflush(stdout);

  gen_corr_gaussian(N_nodes, sqrt_diag, sqrt_recip, sqrt_conv, 
		    sqrt_div, sqrt_chain, sqrt_noshare, W_gaus, rng, algorithm);

  cout << "done\n";
  cout.flush();
  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t %s step3 (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  fflush(stdout);

  ////////////////////////////////////////////////////////////
  // Optional step 4: Calculate the covariance structure
  // of the Gaussian matrix 
  // Then, one can check program output to see if the
  // Gaussian matrix was generated correctly
  ////////////////////////////////////////////////////////////

  if(calc_covs) {

   cout << "Calculating correlations...";
   cout.flush();

   double sigma;
   double cov_recip, cov_conv, cov_div, cov_chain, cov_noshare;
   
   calc_gaus_covs(W_gaus, N_nodes,
		   sigma,cov_recip,
		   cov_conv, cov_div,
		   cov_chain,cov_noshare);

    cout << "done\n";
    cout << "sigma = " << sigma << ", cov_recip = " << cov_recip
	 << ", cov_conv = " << cov_conv
	 << ", cov_div = " << cov_div
	 << ", cov_chain = " << cov_chain
	 << ", cov_noshare = " << cov_noshare
	 << "\n";
    cout.flush();
    gettimeofday(&T1,NULL);
    timersub(&T1,&T0, &dT);
    fprintf(stdout,"t=%li.%06li\t %s step4 (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
    fflush(stdout);

  }

  return W_gaus;

}


////////////////////////////////////////////////////////////////////
// auxilliary functions
////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
// gen_corr_gaussian
// Generate correlated Gaussian given the square root of the 
// covariance matrix determined by sqrt_pars
////////////////////////////////////////////////////////////
int gen_corr_gaussian(const int N_nodes, double sqrt_diag, double sqrt_recip,
		      double sqrt_conv, double sqrt_div, double sqrt_chain,
              double sqrt_noshare, gsl_matrix_float *thevars, gsl_rng *rng, const char *algorithm) {

  struct timeval localT0,T1,dT;

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s init (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  fflush(stdout);

  gsl_matrix_float_set_zero(thevars);
  float *thedata = thevars->data;
  size_t tda=thevars->tda;

  // for speed we'll access the gsl_matrix entries directly
  // from the data structure
  // we need to know the tda (trailing dimension) of the data structure
  // then thedata[i*tda+j] is entry (i,j) from the matrix
  // generate N_nodes*(N_nodes-1) independent Gaussians
  // then multipy by square root of covariance matrix 
  // determined by sqrt
  
  
  double row_sums[N_nodes];
  double column_sums[N_nodes];
  for(int i=0; i<N_nodes; i++)
    row_sums[i]=column_sums[i]=0.0;
  double matrix_sum = 0.0;

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s init_done (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  fflush(stdout);

  double a = (sqrt_diag-sqrt_conv-sqrt_div+sqrt_noshare);
  double b = (sqrt_recip-2.0*sqrt_chain+sqrt_noshare);
  int BLKSIZE=1;

  gettimeofday(&localT0,NULL);

if ((algorithm==NULL) || !strncmp(algorithm,"S",1)) {

  for (int i = 0; i < N_nodes; i++) {
    size_t i_tda = i * tda;
    for (int j = 0; j < N_nodes; j++) {
      // no connection from node onto itself
      if(j==i)
          continue;
      
      double gaus_ind= gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[i_tda + j] += gaus_ind * a;

      // add reciprocal contribution
      thedata[j*tda + i] += gaus_ind * b;

      row_sums[i]    += gaus_ind;
      column_sums[j] += gaus_ind;
    }

    if ( (i % (BLKSIZE<<4)) == 0) {
      gettimeofday(&T1,NULL);
      timersub(&T1,&localT0, &dT);
      double t1 = dT.tv_sec+dT.tv_usec*1e-6;
      double test = t1*N_nodes/(i+1);
      fprintf(stdout,"S %6i t=%f/%f %s (%s line %i)\n", i+1, t1, test, __func__, __FILE__, __LINE__);
      fflush(stdout);
    }
  }

}
else if (!strncmp(algorithm,"T",1)) {

  for (int i = 0; i < N_nodes; i++) {
    size_t i_tda = i * tda;
    for (int j = i+1; j < N_nodes; j++) {
      // no connection from node onto itself

      double gaus_ind;

      /**** upper triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[i_tda + j] += gaus_ind*a;

      // add reciprocal contribution
      thedata[j*tda + i] += gaus_ind*b;

      row_sums[i]    += gaus_ind;
      column_sums[j] += gaus_ind;



      /**** lower triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[j*tda + i] += gaus_ind*a;

      // add reciprocal contribution
      thedata[i_tda + j] += gaus_ind*b;


      row_sums[j]    += gaus_ind;
      column_sums[i] += gaus_ind;

    }
    if ( (i % (BLKSIZE<<4)) == 0) {
      gettimeofday(&T1,NULL);
      timersub(&T1,&localT0, &dT);
      double t1 = dT.tv_sec+dT.tv_usec*1e-6;
      double test = t1*N_nodes*N_nodes/(2 * N_nodes * (i+1) - (i+1)*(i+1));
      fprintf(stdout,"%c %6i t=%f/%f %s (%s line %i)\n", algorithm[0], i+1, t1, test, __func__, __FILE__, __LINE__);
      fflush(stdout);
    }
  }
}

else if (algorithm[0]=='B') {
  BLKSIZE=atoi(algorithm+1);
  int i=0;
  for (int ii = 0;  ii < N_nodes; ii+=BLKSIZE) {
  for (int jj = ii; jj < N_nodes; jj+=BLKSIZE) {
  for (i = ii; i < min(N_nodes, ii+BLKSIZE); i++) {
      size_t i_tda = i * tda;
  for (int j = max(i+1,jj); j < min(N_nodes, jj+BLKSIZE); j++) {
      // no connection from node onto itself
      size_t j_tda = j * tda;


      double gaus_ind;

      /**** upper triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[i_tda + j] += gaus_ind*a;

      // add reciprocal contribution
      thedata[j_tda + i] += gaus_ind*b;

      row_sums[i]    += gaus_ind;
      column_sums[j] += gaus_ind;



      /**** lower triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[j_tda + i] += gaus_ind*a;

      // add reciprocal contribution
      thedata[i_tda + j] += gaus_ind*b;


      row_sums[j]    += gaus_ind;
      column_sums[i] += gaus_ind;

    }
    }
    }

    if ( (ii % (BLKSIZE<<4)) == 0) {
      gettimeofday(&T1,NULL);
      timersub(&T1,&localT0, &dT);
      double t1 = dT.tv_sec+dT.tv_usec*1e-6;
      double test = t1*(size_t)N_nodes*N_nodes/(2.0*N_nodes*i-i*i );
      fprintf(stdout,"%s %5i t=%f/%f %s (%s line %i)\n", algorithm, i, t1, test, __func__, __FILE__, __LINE__);
      fflush(stdout);
    }
  }
}

else if (algorithm[0]=='P') {

  const int MAXBLKSIZE=32;
  BLKSIZE=atoi(algorithm+1);
  if (BLKSIZE>MAXBLKSIZE) exit(-1);

  size_t N = N_nodes/BLKSIZE + (N_nodes%BLKSIZE > 0);

  double s =  gsl_ran_flat (rng, 0.0, (double)N_nodes * N_nodes);
  size_t progress0=0;
  size_t progress1=N_nodes;


#pragma omp parallel for shared(thedata,row_sums,column_sums)
  for (size_t kk = 0;  kk < N*N; kk++) {
	int ii = (kk / N) * BLKSIZE;
	int jj = (kk % N) * BLKSIZE;

	double local_row_sums[MAXBLKSIZE];
	double local_column_sums[MAXBLKSIZE];
	for (int i=0; i<BLKSIZE; i++) {
		local_row_sums[i]=0;
		local_column_sums[i]=0;
	}

	gsl_rng *local_rng = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (local_rng, kk+(int)floor(s));

  for (int i = ii; i < min(N_nodes, ii+BLKSIZE); i++) {
      size_t i_tda = i * tda;
  for (int j = jj; j < min(N_nodes, jj+BLKSIZE); j++) {

      // no connection from node onto itself
	if (i==j) continue;

      double gaus_ind;

      /**** upper triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(rng);

      // add diagonal contribution
      thedata[i_tda + j] += gaus_ind*a;

      // add reciprocal contribution
      thedata[j*tda + i] += gaus_ind*b;

      local_row_sums[i-ii]    += gaus_ind;
      local_column_sums[j-jj] += gaus_ind;

    }
    }
      gsl_rng_free (local_rng);

//#pragma omp barrier
	// reduction step
	for (int i=0; i<BLKSIZE; i++) {
		row_sums[i+ii]   +=local_row_sums[i];
		column_sums[i+jj]+=local_column_sums[i];
	}

/*
	progress0+=1;
	if (progress0 > progress1) {
		progress1 += N_nodes*BLKSIZE;
		gettimeofday(&T1,NULL);
		timersub(&T1,&localT0, &dT);
		double t1 = dT.tv_sec+dT.tv_usec*1e-6;
		double test = (t1*N_nodes)/(2.0*progress0);
		fprintf(stdout,"%s %5i t=%f/%f (%li/%li) %s (%s line %i)\n", algorithm, N_nodes, t1, test,progress0,progress1, __func__, __FILE__, __LINE__);
		fflush(stdout);
	}
*/
  }


}

else if (algorithm[0]=='X') {

  const int MAXBLKSIZE=32;
  BLKSIZE=atoi(algorithm+1);
  if (BLKSIZE>MAXBLKSIZE) exit(-1);

  size_t N = N_nodes/BLKSIZE + (N_nodes%BLKSIZE > 0);

  double s =  gsl_ran_flat (rng, 0.0, (double)N_nodes * N_nodes);
  size_t progress0=0;
  size_t progress1=N_nodes;


#pragma omp parallel for shared(thedata,row_sums,column_sums)
  for (size_t kk = 0;  kk < N*N; kk++) {
	int ii = (kk / N) * BLKSIZE;
	int jj = (kk % N) * BLKSIZE;
	if (jj<ii) continue;

	double local_row_sums_i[MAXBLKSIZE];
	double local_column_sums_i[MAXBLKSIZE];
	double local_row_sums_j[MAXBLKSIZE];
	double local_column_sums_j[MAXBLKSIZE];
	for (int i=0; i<BLKSIZE; i++) {
		local_row_sums_i[i]=0;
		local_column_sums_i[i]=0;
		local_row_sums_j[i]=0;
		local_column_sums_j[i]=0;
	}

	gsl_rng *local_rng = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (local_rng, kk+(int)floor(s));

  for (int i = ii; i < min(N_nodes, ii+BLKSIZE); i++) {
      size_t i_tda = i * tda;
  for (int j = max(i+1,jj); j < min(N_nodes, jj+BLKSIZE); j++) {
      // no connection from node onto itself
      size_t j_tda = j * tda;


      double gaus_ind;

      /**** upper triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(local_rng);

      // add diagonal contribution
      thedata[i_tda + j] += gaus_ind*a;

      // add reciprocal contribution
      thedata[j_tda + i] += gaus_ind*b;

      local_row_sums_i[i-ii]    += gaus_ind;
      local_column_sums_j[j-jj] += gaus_ind;



      /**** lower triangle matrix elements ****/
      gaus_ind = gsl_ran_ugaussian(local_rng);

      // add diagonal contribution
      thedata[j_tda + i] += gaus_ind*a;

      // add reciprocal contribution
      thedata[i_tda + j] += gaus_ind*b;


      local_row_sums_j[j-jj]    += gaus_ind;
      local_column_sums_i[i-ii] += gaus_ind;

    }
    }
      gsl_rng_free (local_rng);

//#pragma omp barrier
	// reduction step
	for (int i=0; i<BLKSIZE; i++) {
		row_sums[i+ii]   +=local_row_sums_i[i];
		column_sums[i+ii]+=local_column_sums_i[i];
	}
	for (int j=0; j<BLKSIZE; j++) {
		row_sums[j+jj]   +=local_row_sums_j[j];
		column_sums[j+jj]+=local_column_sums_j[j];
	}

/*
	progress0+=1;
	if (progress0 > progress1) {
		progress1 += N_nodes*BLKSIZE;
		gettimeofday(&T1,NULL);
		timersub(&T1,&localT0, &dT);
		double t1 = dT.tv_sec+dT.tv_usec*1e-6;
		double test = (t1*N_nodes)/(2.0*progress0);
		fprintf(stdout,"%s %5i t=%f/%f (%li/%li) %s (%s line %i)\n", algorithm, N_nodes, t1, test,progress0,progress1, __func__, __FILE__, __LINE__);
		fflush(stdout);
	}
*/
  }


}

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s matrix_sum_start (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  // Check danger of Rounding errors, do a SUM over column_sums or row_sums instead
  for (int i=0; i<N_nodes; i++) {
    matrix_sum += row_sums[i] + column_sums[i];
  }
  matrix_sum *= 0.5;

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s matrix_sum_done (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);

#pragma omp parallel for shared(thedata)
  for (int i=0; i<N_nodes; i++) {
    size_t i_tda=i*tda;
    for (int j=0; j<N_nodes; j++) {
      // no connection from node onto itself
      if(i==j) continue;
      
      thedata[i_tda+j]+=(sqrt_conv-sqrt_noshare)*row_sums[i]+
	(sqrt_div-sqrt_noshare)*column_sums[j]+
	(sqrt_chain-sqrt_noshare)*(row_sums[j]+column_sums[i])
	+sqrt_noshare*matrix_sum;
    }
  }

  gettimeofday(&T1,NULL);
  timersub(&T1,&T0, &dT);
  fprintf(stdout,"t=%li.%06li\t%s done (%s line %i)\n",dT.tv_sec,dT.tv_usec,__func__,__FILE__,__LINE__);
  return 0;
}


////////////////////////////////////////////////////////////
// calc_gaus_covs
// calculate the covariance structure of the gaussian
// random matrix
////////////////////////////////////////////////////////////

int calc_gaus_covs(gsl_matrix *W_gaus, int N_nodes,
		   double &sigma, double &cov_recip,
		   double &cov_conv, double &cov_div,
		   double &cov_chain, double &cov_noshare) {

  // calc covariances of Gaussian (assume everything mean zero)

  sigma=0.0;
  cov_recip=0.0;
  cov_conv=0.0;
  cov_div=0.0;
  cov_chain=0.0;
  cov_noshare=0.0;

  for(int i=0; i<N_nodes; i++) {
    for(int j=0; j<N_nodes; j++) {
      if(i==j)
	continue;
	  
      double w_ij = gsl_matrix_get(W_gaus,i,j);
      sigma += w_ij*w_ij;
	  
      cov_recip += w_ij * gsl_matrix_get(W_gaus,j,i);
	  
      for(int k=0; k<N_nodes; k++) {
	if(k==i || k==j)
	  continue;
	    
	cov_conv += w_ij * gsl_matrix_get(W_gaus, i,k);
	cov_div += w_ij * gsl_matrix_get(W_gaus, k,j);
	cov_chain += w_ij * gsl_matrix_get(W_gaus, j,k);
	cov_chain += w_ij * gsl_matrix_get(W_gaus, k,i);

	// subsample edges that don't share a node
	if(k+1 <N_nodes && k+1 != i && k+1 !=j) 
	  cov_noshare += w_ij * gsl_matrix_get(W_gaus, k, k+1);
	
      }
    }
  }

  sigma /= N_nodes*(N_nodes-1.0);
  sigma = sqrt(sigma);
  cov_recip /= N_nodes*(N_nodes-1.0);
  cov_conv /= N_nodes*(N_nodes-1.0)*(N_nodes-2.0);
  cov_div /= N_nodes*(N_nodes-1.0)*(N_nodes-2.0);
  cov_chain /= 2*N_nodes*(N_nodes-1.0)*(N_nodes-2.0);
  cov_noshare /= (N_nodes-1.0)*(N_nodes-2.0)*(N_nodes-3.0);

  return 0;

}
