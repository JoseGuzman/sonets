#include <gsl/gsl_matrix_float.h>

int calc_phat_alphahat_1p(gsl_matrix_float *W, int N_nodes,
			  double &phat,
			  double &alphahat_recip,
			  double &alphahat_conv,
			  double &alphahat_div,
			  double &alphahat_chain,
			  double &alphahat_other);
int calc_N_motifs(gsl_matrix_float *W, int N_nodes,
		  double &N_edge, double &N_recip,
		  double &N_conv, double &N_div, double &N_chain,
		  double &N_other);
