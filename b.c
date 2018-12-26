
#include <stdio.h>
#include "mpi.h"

#include <sys/timeb.h>
#include <stdlib.h>
#include <math.h>

const int N = 100;
const int M = 25;
double h = 1./(N + 1);
double eps = 1e-13;
int NP = 4;

double u2[N+2][N+2];

double f1(double y) {
	return 1. - y*y;
}

double f3(double y) {
	return sin(y);
}

double f2(double y) {
	return y*y*y;
}

double f4(double y) {
	return 1. - y*y;
}

double F(double x, double y) {
	return x*y;
}


int main(int argc, char ** argv) {

int myrank, ranksize;
	int i;
	MPI_Status status;

	MPI_Init (&argc, &argv);       /* initialize MPI system */
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);    /* my place in MPI system */
	MPI_Comm_size (MPI_COMM_WORLD, &ranksize);  /* size of MPI system */

	double start = MPI_Wtime();

	for (int i=0; i < N+2; i++)
		for (int j=0; j < N+2; j++)
			u2[i][j] = 0;

	for (int i=0; i < N+2; i++) {
			u2[0][i] = f1(h*i);
			u2[N+1][i] = f3(h*i);
			u2[i][0] = f4(i*h);
			u2[i][N+1] = f2(i*h);
	}

	int ni = 0;
	while (1) {
			ni++;
			double dmax = 0.;
			for (int i=1; i<N+1; i++)
			for (int j=1; j<N+1; j++) {
				double temp = u2[i][j];
				u2[i][j] = 0.25 * (u2[i-1][j] + u2[i+1][j] + u2[i][j-1] + u2[i][j+1] - h*h*F(i*h, j*h));
				if (fabs(temp - u2[i][j]) > dmax)
					dmax = fabs(temp - u2[i][j]);
			}
			if (dmax < eps) break;
	}

	double end = MPI_Wtime();

	fprintf(stderr, "Calculating time for single process %f and number of iterations %d\n", end - start, ni);

	freopen("output2.txt", "w", stdout);

	for (int i=1; i<N+1; i++)
		for (int j=1; j<N+1; j++)
			printf("%f %f %f\n", i*h, j*h, u2[i][j]);

	MPI_Finalize ();
	return 0;
}
	