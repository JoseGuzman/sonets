#include <gsl/gsl_blas.h>

#include "calc_stats_1p.hpp"

////////////////////////////////////////////////////////////////////
// calc_phat_alphahat_1p
//
// Calculate first and second order connectivity statistics from a 
// network adjacency matrix W with N_node nodes
////////////////////////////////////////////////////////////////////


int calc_phat_alphahat_1p(gsl_matrix_float *W, int N_nodes,
			  double &phat,
			  double &alphahat_recip,
			  double &alphahat_conv,
			  double &alphahat_div,
			  double &alphahat_chain,
			  double &alphahat_other) {

  double N_edge, N_recip, N_conv, N_div, N_chain, N_other;
  
  calc_N_motifs(W, N_nodes, N_edge, N_recip, N_conv, N_div, N_chain, N_other);
  
  double c1 = N_nodes*(N_nodes-1.0);
  double c2 = N_nodes*(N_nodes-1.0)*(N_nodes-2.0)/2.0;
  double c3 = N_nodes*(N_nodes-1.0)*(N_nodes-2.0)*(N_nodes-3.0)/2.0;
  phat =N_edge/c1;
  alphahat_recip =N_recip/(c1/2.0*phat*phat)-1;
  alphahat_conv =  N_conv/(c2*phat*phat)-1;
  alphahat_div = N_div/(c2*phat*phat)-1;
  alphahat_chain = N_chain/(2*c2*phat*phat)-1;
  alphahat_other =  N_other/(c3*phat*phat)-1;

  return 0;

}

////////////////////////////////////////////////////////////////////
// calc_N_motifs
//
// Calculate number of edges and two edges motifs from a 
// network adjacency matrix W with N_node nodes
////////////////////////////////////////////////////////////////////

int calc_N_motifs(gsl_matrix_float *W, int N_nodes,
	       double &N_edge, double &N_recip,
	       double &N_conv, double &N_div, double &N_chain,
	       double &N_other) {

  float *thedata = W->data;
  size_t tda = W->tda;

  double row_sums[N_nodes];
  double column_sums[N_nodes];
  double W_square_trace=0.0;
  N_edge=0;
  for(int i=0; i<N_nodes;i++)
    row_sums[i]=column_sums[i]=0.0;
  for(int i=0; i<N_nodes;i++) {
    size_t i_tda = i*tda;
    for(int j=0; j<N_nodes;j++) {
      double temp=thedata[i_tda+j];
      row_sums[i]+=temp;
      column_sums[j]+=temp;
      W_square_trace += temp*thedata[j*tda+i];
      N_edge += temp;
    }
  }

  N_recip = W_square_trace/2;
  
  // N_chain, N_conv, and N_div
  N_chain = 0.0;
  N_conv = 0.0;
  N_div = 0.0;
  for (int i=0;i<N_nodes;i++){
    N_conv += row_sums[i]*row_sums[i];
    N_div += column_sums[i]*column_sums[i];
    N_chain += row_sums[i]*column_sums[i];
  }
  N_chain -= W_square_trace;
  N_conv -= N_edge;
  N_div -= N_edge;
  N_conv /= 2;
  N_div /= 2;
  
  N_other=N_edge*(N_edge-1.0)/2.0- (N_conv+N_div+N_chain+N_recip);
  
  return 0;

}

