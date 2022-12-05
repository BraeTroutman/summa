#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cassert>

using namespace std;

const bool DEBUG = false;

// initialize matrix and vectors (A is mxn, x is xn-vec)
void init_rand(double* a, int m, int n);
// local matvec: y = y+A*x, where A is m x n
void local_gemv(double* A, double* x, double* y, int m, int n);

int main(int argc, char** argv) {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int nProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank*12345);

    // Read dimensions and processor grid from command line arguments
    if(argc != 4) {
        cerr << "Usage: ./a.out m b identity[yes|no])" << endl;
        return 1;
    }

    int m, b;
    m = atoi(argv[1]);
    b = atoi(argv[2]);
	bool IDENT = !strcmp(argv[3], "yes");
    
    int p = sqrt(nProcs);
    if (p*p != nProcs) {
        cerr << "number of processes must be a perfect square" << endl;
        return 1;
    }

    if(m % p || m % nProcs) {
        cerr << "Processor grid doesn't divide rows and columns evenly" << endl;
        return 1;
    }

    // Set up row and column communicators
    int ranki = rank % p; // proc row coordinate
    int rankj = rank / p; // proc col coordinate
    
    // Create row and column communicators using MPI_Comm_split
    MPI_Comm row_comm, col_comm;
   	MPI_Comm_split(MPI_COMM_WORLD, rankj, rank, &col_comm);
	MPI_Comm_split(MPI_COMM_WORLD, ranki, rank, &row_comm);

    // Check row and column communicators and proc coordinates
    int rankichk, rankjchk;
    MPI_Comm_rank(row_comm,&rankjchk);
    MPI_Comm_rank(col_comm,&rankichk);

    if(ranki != rankichk || rankj != rankjchk) {
        cerr << "Processor ranks are not as expected, check row and column communicators" << endl;
        return 1;
    }

    // Initialize matrices and vectors
    int mloc = m / p;     // number of rows of local matrix
    int nloc = m / p;     // number of cols of local matrix
    
    double* Alocal = new double[mloc*nloc];
    double* Blocal = new double[mloc*nloc];
    double* Clocal = new double[mloc*nloc];

    double* Atemp = new double[b*mloc];
    double* Btemp = new double[b*mloc];

   	init_rand(Alocal, mloc, nloc);
	if (IDENT) {
		for (int i = 0; i < mloc; i++) {
			for (int j = 0; j < nloc; j++) {
				if (i + ranki*mloc == j + rankj*nloc) {
					Blocal[i+j*mloc] = 1;
				} else {
					Blocal[i+j*mloc] = 0;
				}
			}
		}
	}
    memset(Clocal,0,mloc*nloc*sizeof(double));

    // start timer
    double start = MPI_Wtime();
    
    // broadcasts and local multiplications
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < m / (b*p); j++) {
            MPI_Bcast(Atemp, mloc*b, MPI_DOUBLE, j, row_comm);
            MPI_Bcast(Btemp, mloc*b, MPI_DOUBLE, j, col_comm);
        }
    }

	// Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start;

    // Print time
    if(!rank) {
        cout << nProcs << ","
			<< m << "," 
			<< p << "," 
			<< total_time << endl;
    }

    // Clean up
    delete [] Alocal;
    delete [] Blocal;
    delete [] Clocal;
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
}

void local_gemv(double* a, double* x, double* y, int m, int n) {
    // order for loops to match col-major storage
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            y[i] += a[i+j*m] * x[j];
        }
    }
}

void init_rand(double* a, int m, int n) {
    // init matrix
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            a[i+j*m] = rand() % 100;
        }
    }
}
