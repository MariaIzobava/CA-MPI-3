
#include <stdio.h>
#include "mpi.h"
#include <sys/timeb.h>
#include <stdlib.h>
#include <math.h>

const int N = 100;
const int M = 25;
double h = 1./(N + 1);
double eps = 1e-13;
double ans[6*N*M + 10][3];
int NP = 4;
int k = 0;

double u[M+2][N+2];

double u2[N+2][N+2];

double f1(double y) {
	return y*y;

}

double f3(double y) {
	return sin(y);
}

double f2(double y) {
	return y*y*y;
}

double f4(double y) {
	return y*y;
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
	for (int i=0; i < M+2; i++)
		for (int j=0; j < N+2; j++)
			u[i][j] = 0;


	if (myrank == 0) {
		for (int i = 1; i < N + 1; i++) {
			u[0][i] = f1(i * h);
		}
	}


	if (myrank == NP-1) {
		for (int i = 1; i < N + 1; i++) {
			u[M+1][i] = f3(i * h);
		}
	}


	for (int i = 1; i < M + 1; i++) {
		u[i][0] = f4((i - 1 + myrank * M) * h);
		u[i][N + 1] = f2((i - 1 + myrank * M) * h);
	}


int kl=0;

	while (1) {
		kl++;
		if (myrank < NP - 1) 
			MPI_Send(u[M], N+2, MPI_DOUBLE, myrank + 1, 98, MPI_COMM_WORLD);

		if (myrank > 0)
			MPI_Recv(u[0], N+2, MPI_DOUBLE, myrank-1, 98, MPI_COMM_WORLD, &status);
		
		MPI_Barrier (MPI_COMM_WORLD);

		if (myrank > 0) 
			MPI_Send(u[1], N+2, MPI_DOUBLE, myrank - 1, 98, MPI_COMM_WORLD);

		if (myrank < NP - 1)
			MPI_Recv(u[M+1], N+2, MPI_DOUBLE, myrank + 1, 98, MPI_COMM_WORLD, &status);
		MPI_Barrier (MPI_COMM_WORLD);
		double dmax = 0.;
		for (int i=1; i<M+1; i++)
			for (int j=1; j<N+1; j++) {
				double temp = u[i][j];
				u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h*h*F(i*h, j*h));
				if (fabs(temp - u[i][j]) > dmax)
					dmax = fabs(temp - u[i][j]);
			}
		
		if (myrank != 0)
			MPI_Send(&dmax, 1, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD);
		else {
			for (int i = 1; i<ranksize; i++) {
				double r;
				MPI_Recv(&r, 1, MPI_DOUBLE, i, 98, MPI_COMM_WORLD, &status);
				if (r > dmax)
					dmax = r;
			}
			//printf("%f\n", dmax);
		}
		MPI_Barrier (MPI_COMM_WORLD);
		MPI_Bcast(&dmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (dmax < eps) break;
		//break;
	
	}

	double end = MPI_Wtime();

	fprintf(stderr, "Calculating time for single process %f %d\n", end - start, kl);

	if (myrank == 0) {
		for (int i=1; i<M+1; i++) {
					for (int j=0; j<N+2; j++) {
						ans[k][0] = i * h;
						ans[k][1] = j * h;
						ans[k][2] = u[i][j];
						k += 1;
					}
			}		

		for (int p=1; p<ranksize; p++) {
			for (int i=1; i<M+1; i++) {
					for (int j=0; j<N+2; j++) {
						double f[3];
						MPI_Recv(f, 3, MPI_DOUBLE, p, 98, MPI_COMM_WORLD, &status);
						ans[k][0] = f[0];
						ans[k][1] = f[1];
						ans[k][2] = f[2];
						k += 1;
					}
			}
		}
		
	} else {
			for (int i=1; i<M+1; i++) {
					for (int j=0; j<N+2; j++) {
						double f[3];
						f[0] = (myrank * M + i) * h;
						f[1] = j * h;
						f[2] = u[i][j];
						MPI_Send(f, 3, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD);
					}
			}
	}

	MPI_Barrier (MPI_COMM_WORLD);


	freopen("output.txt", "w", stdout);
	if (myrank == 0) {
		for (int i=0; i<k; i++){
				printf("%f %f %f\n", ans[i][0], ans[i][1], ans[i][2]);
			}
	}

	MPI_Finalize ();
	return 0;
}