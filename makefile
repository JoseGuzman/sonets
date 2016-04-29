CC = g++
C_OPTIMIZE_SWITCH = -O3 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
LIBS = -lgsl -lgslcblas -lgomp

CFLAGS = -Wall ${C_OPTIMIZE_SWITCH} -fopenmp

## Define sparse output format
#  SPARSE_MM   clear text ascii format
#  SPARSE_COO  coordinate format, binary
#  SPARSE_CSR  compressed row storage, most space efficient, default
#CFLAGS += -DSPARSE_MM
#CFLAGS += -DSPARSE_COO
#CFLAGS += -DSPARSE_CSR


run_secorder: run_secorder.o secorder_rec_1p.o calc_sqrtcov_rec_1p.o calc_rhos.o calc_stats_1p.o
	${CC} run_secorder.o secorder_rec_1p.o calc_sqrtcov_rec_1p.o calc_rhos.o calc_stats_1p.o -o $@ ${LIBS}

run_secorder_2p: run_secorder_2p.o secorder_rec_2p.o calc_sqrtcov_rec_2p.o calc_rhos.o calc_stats_2p.o
	${CC} run_secorder_2p.o secorder_rec_2p.o calc_sqrtcov_rec_2p.o calc_rhos.o calc_stats_2p.o -o run_secorder_2p ${LIBS}

run_secorder.o: secorder_rec_1p.hpp calc_stats_1p.hpp
run_secorder_2p.o: secorder_rec_2p.hpp calc_stats_2p.hpp
secorder_rec_1p.o: secorder_rec_1p.hpp calc_sqrtcov_rec_1p.hpp calc_rhos.hpp
secorder_rec_2p.o: secorder_rec_2p.hpp calc_sqrtcov_rec_2p.hpp calc_rhos.hpp calc_stats_2p.hpp
calc_sqrtcov_rec_1p.o: calc_sqrtcov_rec_1p.hpp
calc_sqrtcov_rec_2p.o: calc_sqrtcov_rec_2p.hpp
calc_rhos.o: calc_rhos.hpp
calc_stats_1p.o: calc_stats_1p.hpp
calc_stats_2p.o: calc_stats_2p.hpp


%.o : %.cpp
	${CC} -c ${CFLAGS} $<

clean:
	rm *.o
