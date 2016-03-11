#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_cdf.h>

#include "calc_sqrtcov_rec_1p.hpp"

using namespace std;

int calc_sqrtcov_given_rhos_large_N
(int N, double sigma, double rho_recip, double rho_conv, 
 double rho_div, double rho_chain,
 double &sqrt_diag, double &sqrt_recip, double &sqrt_conv,
 double &sqrt_div, double &sqrt_chain);
int calc_sqrtcov_given_rhos_refine
(int N, double sigma, double rho_recip, double rho_conv, 
 double rho_div, double rho_chain, double rho_noshare,
 double &sqrt_diag, double &sqrt_recip, double &sqrt_conv,
 double &sqrt_div, double &sqrt_chain, double &sqrt_noshare);


//////////////////////////////////////////////////////////////////
// calc_sqrtcov_given_rhos
//
// calculate parameters determining the square root of 
// the gaussian covariance matrix
// given the paramters rho determining the covariance structure
//
// Since not all combinations of rho are valid
// it is important to be able to test for valid parameters.
// This is simple to do in the large N limit, as we can reduce
// most of the calculation to taking the square root 
// of a small matrix. 
// Therefore, first calculate answer in that limit.
// Then use that estimate as an initial estimate for the 
// numerical routine to calculate for the general case.
//////////////////////////////////////////////////////////////////
int calc_sqrtcov_given_rhos
(int N, double p, double rho_recip, double rho_conv, 
 double rho_div, double rho_chain, double rho_noshare,
 double &sqrt_diag, double &sqrt_recip, double &sqrt_conv,
 double &sqrt_div, double &sqrt_chain, double &sqrt_noshare) {

  // standard deviation of each component of the Gaussian
  // so that probability will be above zero will be p
  double sigma = 1.0/gsl_cdf_ugaussian_Qinv(p);

  // first estimate standard deviations in the limit
  // of a large network
  int status = calc_sqrtcov_given_rhos_large_N
    (N, sigma, rho_recip, rho_conv, rho_div, rho_chain,
     sqrt_diag, sqrt_recip, sqrt_conv, sqrt_div, sqrt_chain);
  sqrt_noshare=0.0;

  if(status)
    return status;
  
  // cout << "Before refine:\n";
  // cout << "sqrt_diag = " << sqrt_diag
  // 	 << ", sqrt_recip = " << sqrt_recip
  // 	 << ", sqrt_conv = " << sqrt_conv
  // 	 << ", sqrt_div = " << sqrt_div
  // 	 << ", sqrt_chain = " << sqrt_chain
  // 	 << "\n";

  if(1) {
    status = calc_sqrtcov_given_rhos_refine
      (N, sigma, rho_recip, rho_conv, rho_div, rho_chain, rho_noshare,
       sqrt_diag, sqrt_recip, sqrt_conv, sqrt_div, sqrt_chain, sqrt_noshare);


    // cout << "After refine:\n";
    // cout << "sqrt_diag = " << sqrt_diag
    // 	 << ", sqrt_recip = " << sqrt_recip
    // 	 << ", sqrt_conv = " << sqrt_conv
    // 	 << ", sqrt_div = " << sqrt_div
    // 	 << ", sqrt_chain = " << sqrt_chain
    // 	 << "\n";

  }

  return status;

}


// calculate the components of the sqrt of the covariance matrix
// assuming that the number of neurons N is large
int calc_sqrtcov_given_rhos_large_N
(int N, double sigma, double rho_recip, double rho_conv, 
 double rho_div, double rho_chain,
 double &sqrt_diag, double &sqrt_recip, double &sqrt_conv,
 double &sqrt_div, double &sqrt_chain) {
  
  // for large N, the equations for sqrt_conv, sqrt_div, and sqrt_chain
  // decouple from the rest.
  
  // moreover, these 3 equations can be written as solving for the
  // square root of a 2x2 covariance matrix
  // Hence one can quickly determine if the equations have a real solution
  // and find that real solution

  const size_t nmat=2;
  gsl_eigen_symmv_workspace *work_eig=gsl_eigen_symmv_alloc(nmat);
  
  gsl_matrix *A = gsl_matrix_alloc(nmat,nmat);    // the 2x2 covariance matrix
  gsl_matrix *sqrtA =gsl_matrix_alloc(nmat,nmat); // its square root
  gsl_vector *evals=gsl_vector_alloc(nmat);       // evals of A
  gsl_matrix *evecs=gsl_matrix_alloc(nmat,nmat);  // evects of A

  gsl_matrix_set(A,0,0, rho_conv);
  gsl_matrix_set(A,1,1, rho_div);
  gsl_matrix_set(A,1,0, rho_chain);

  gsl_matrix_scale(A, sigma*sigma);

  // to calculate square root of A
  // 1. take it's eigen decomposition
  // 2. take the square root of its eigenvalues
  // 3. reconstruct with new eigenvalues to get a square root of A
  
  gsl_eigen_symmv(A, evals, evecs, work_eig);
  
  for(size_t i=0; i<nmat; i++) {
    double the_eval = gsl_vector_get(evals,i);
    if(the_eval <= 0) {
      if(the_eval > -1E-12) {
	// allow eigenvalues to be slightly negative due to
	// roundoff error
	the_eval=0;
      }
      else {
	// if have a negative eigenvalue, can't take square root
	// system of equations does not have a real solution
	// (at least in limit of large N)
	cout << "Found a negative eval(" << i <<")=" << the_eval << "\n";
	gsl_eigen_symmv_free(work_eig);
	gsl_matrix_free(A);
	gsl_matrix_free(sqrtA);
	gsl_matrix_free(evecs);
	gsl_vector_free(evals);
	return -1;
      }
    }
    
    // scale eigenvector by fourth root of eval so 
    // reconstruction with be based on square root of eval
    gsl_vector_view col = gsl_matrix_column(evecs,i);
    gsl_vector_scale(&col.vector, sqrt(sqrt(the_eval)));
  }
  
  // reconstruct to get sqrt A
  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,evecs,evecs,0,sqrtA);
  
  // undo scaling to get elements of the square root of 
  // original covariance matrix
  // obtain approximate solutions for sqrt_conv, sqrt_div, and sqrt_chain
  gsl_matrix_scale(sqrtA,1.0/sqrt(N));
  sqrt_conv=gsl_matrix_get(sqrtA,0,0);
  sqrt_div=gsl_matrix_get(sqrtA,1,1);
  sqrt_chain=gsl_matrix_get(sqrtA,1,0);
  
  // can solve for sqrt_recip and sqrt_diag entries just involing nrn_type
  double temp1=sigma*sigma-(N-2.0)*(gsl_pow_2(sqrt_conv)
				    + gsl_pow_2(sqrt_div)
				    + 2.0*gsl_pow_2(sqrt_chain));
  double temp2 = sigma*sigma*rho_recip
    -2*(N-2.0)*(sqrt_conv+sqrt_div)*sqrt_chain;
  
  // calculate sqrt_diag
  if(fabs(temp1) >= fabs(temp2)) {
    sqrt_diag = sqrt((temp1+sqrt(temp1*temp1-temp2*temp2))/2.0);
  }
  else {
    // if can't get real solution to sqrt_diag, original system did
    // not have a real solution (at least for large N)
    cout << "Can't calculate sqrt_diag\n";
    cout << "temp1=" << temp1
         << ", temp2=" << temp2
         << ", sigma=" << sigma
         << ", sqrt_conv=" << sqrt_conv
         << ", sqrt_div=" << sqrt_div
         << ", sqrt_chain=" << sqrt_chain
         << ", rho_recip=" << rho_recip << "\n";

    gsl_eigen_symmv_free(work_eig);
    gsl_matrix_free(A);
    gsl_matrix_free(sqrtA);
    gsl_matrix_free(evecs);
    gsl_vector_free(evals);
    return -1;
  }
  
  // calculate sqrt_recip
  sqrt_recip = temp2/(2.0*sqrt_diag);


  gsl_eigen_symmv_free(work_eig);
  gsl_matrix_free(A);
  gsl_matrix_free(sqrtA);
  gsl_matrix_free(evecs);
  gsl_vector_free(evals);

  return 0;


}



// structure to hold the parameters for 
// the numerical routine calculating sqrt in general case
struct onepop_params
{
  double sigma;
  double rho_recip;
  double rho_conv;
  double rho_div;
  double rho_chain;
  double rho_noshare;
  int N;
};



// function to solve in order to determine
// square root of covariance
int onepop_f(const gsl_vector * x, void *params,
	    gsl_vector * f) {

  double sigma = ((struct onepop_params *) params)->sigma;
  double rho_recip = ((struct onepop_params *) params)->rho_recip;
  double rho_conv = ((struct onepop_params *) params)->rho_conv;
  double rho_div = ((struct onepop_params *) params)->rho_div;
  double rho_chain = ((struct onepop_params *) params)->rho_chain;
  double rho_noshare = ((struct onepop_params *) params)->rho_noshare;
  int N= ((struct onepop_params *) params)->N;

  double a, b, c, d, e, ff;
  
  a = gsl_vector_get(x,0);
  b = gsl_vector_get(x,1);
  c = gsl_vector_get(x,2);
  d = gsl_vector_get(x,3);
  e = gsl_vector_get(x,4);
  ff= gsl_vector_get(x,5);
  
  double f_diag, f_recip, f_conv, f_div, f_chain, f_noshare;
  double sigma2=sigma*sigma;

  f_diag =  gsl_pow_2(a) + gsl_pow_2(b) 
    + (N-2.0)*(gsl_pow_2(c)+gsl_pow_2(d) + 2*gsl_pow_2(e))
    + (N-2.0)*(N-3.0)*gsl_pow_2(ff)
    - sigma2;

  f_recip =  2*a*b +2*(N-2.0)*(c*e + d*e)-sigma2*rho_recip
    + (N-2.0)*(N-3.0)*gsl_pow_2(ff);

  f_conv =  2*a*c+2*e*(b+d)+(N-3.0)*(gsl_pow_2(c) + gsl_pow_2(e) + 2*(d+e)*ff)
    + (N-3.0)*(N-4.0)*gsl_pow_2(ff)
    - sigma2*rho_conv;
  
  f_div = 2*a*d+ 2*e*(b+c)+(N-3.0)*(gsl_pow_2(d) + gsl_pow_2(e) + 2*(c+e)*ff)
    + (N-3.0)*(N-4.0)*gsl_pow_2(ff)
    - sigma2*rho_div;

  f_chain = 2*a*e + b*(c +d) + c*d + gsl_pow_2(e) +(N-3.0)*e*(c+d)
    + (N-3.0)*ff*(c+d+2*e) + (N-3.0)*(N-4.0)*gsl_pow_2(ff)
    -sigma2*rho_chain;
  
  f_noshare = 2*ff*(a+b)+2*e*(c+d)+2*c*d+2*gsl_pow_2(e)
    + 2*(N-4.0)*ff*(c+d+2*e) + (N-4.0)*(N-5.0)*gsl_pow_2(ff)
    - sigma2*rho_noshare;
  
  gsl_vector_set(f,0,f_diag);
  gsl_vector_set(f,1,f_recip);
  gsl_vector_set(f,2,f_conv);
  gsl_vector_set(f,3,f_div);
  gsl_vector_set(f,4,f_chain);
  gsl_vector_set(f,5,f_noshare);


  return GSL_SUCCESS;
}


// calculate the components of the sqrt of the covariance matrix
// in the general case
// Since this is a numerical root finding calculation,
// it depends on a good initial guess, which was found
// from the large N limit and passed as the
// initial values of the sqrt_pars
int calc_sqrtcov_given_rhos_refine
(int N, double sigma, double rho_recip, double rho_conv, 
 double rho_div, double rho_chain, double rho_noshare,
 double &sqrt_diag, double &sqrt_recip, double &sqrt_conv,
 double &sqrt_div, double &sqrt_chain, double &sqrt_noshare) {


  
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;
  
  int status;
  const size_t n = 6;
  struct onepop_params p = { sigma,rho_recip, 
  			    rho_conv, rho_div,
  			     rho_chain, rho_noshare, N};
                 
  gsl_multiroot_function f = {&onepop_f, n, &p};


  // set initial state based on the sqrt parameters passed into the function
  gsl_vector *x = gsl_vector_alloc (n);

  gsl_vector_set(x,0,sqrt_diag);
  gsl_vector_set(x,1,sqrt_recip);
  gsl_vector_set(x,2,sqrt_conv);
  gsl_vector_set(x,3,sqrt_div);
  gsl_vector_set(x,4,sqrt_chain);
  gsl_vector_set(x,5,sqrt_noshare);

  
  T = gsl_multiroot_fsolver_hybrids;
  s = gsl_multiroot_fsolver_alloc (T, n);
  gsl_multiroot_fsolver_set (s, &f, x);

  int iter=0;
  do
    {
      iter++;
      status = gsl_multiroot_fsolver_iterate (s);
      
      if (status)   /* check if solver is stuck */
	break;
      
      status =
	gsl_multiroot_test_residual (s->f, 1e-7);
    }
  while (status == GSL_CONTINUE && iter < 1000);
  
  // cout << "After " << iter << " iterations of solver, square root refine status = "
  //      << gsl_strerror(status) << "\n";

  // if had problems, don't modify values of sqrt parameter
  if(status) {
    gsl_multiroot_fsolver_free(s);
    gsl_vector_free(x);
    return(status);
  }

  // else set sqrt parameters to result of function
  sqrt_diag = gsl_vector_get(s->x,0);
  sqrt_recip = gsl_vector_get(s->x,1);
  sqrt_conv = gsl_vector_get(s->x,2);
  sqrt_div = gsl_vector_get(s->x,3);
  sqrt_chain = gsl_vector_get(s->x,4);
  sqrt_noshare=gsl_vector_get(s->x,5);

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);

  return 0;
}
