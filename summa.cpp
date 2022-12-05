#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <mpi.h>
#include <cassert>

using namespace std;

const bool DEBUG = true;

// initialize matrix and vectors (A is mxn, x is xn-vec)
void init_rand(double* a, int m, int n);
// local matvec: y = y+A*x, where A is m x n
void local_gemv(double* A, double* x, double* y, int m, int n);
void local_gemm(double* a, double* b, double* c, int m, int n);

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
   
    // column-wise multiplicand
    double* Alocal = new double[mloc*nloc];
    // row-wise multiplier
    double* Blocal = new double[mloc*nloc];
    // column-wise product
    double* Clocal = new double[mloc*nloc];
    
    // n/sqrt(p) by b temporary matrix
    double* Atemp = new double[b*mloc];
    // b by n/sqrt(p) temporary matrix
    double* Btemp = new double[b*mloc];

   	init_rand(Alocal, mloc, nloc);
	if (IDENT) {
		for (int i = 0; i < nloc; i++) {
			for (int j = 0; j < mloc; j++) {
				if (i + ranki*mloc == j + rankj*nloc) {
					Blocal[i*nloc+j] = 1;
				} else {
					Blocal[i*nloc+j] = 0;
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
            memcpy(Atemp, Alocal+b*j*mloc, b*mloc*sizeof(double));
            memcpy(Btemp, Blocal+b*j*nloc, b*nloc*sizeof(double));
            MPI_Bcast(Atemp, b*mloc, MPI_DOUBLE, i, row_comm);
            MPI_Bcast(Btemp, b*nloc, MPI_DOUBLE, i, col_comm);
            local_gemm(Atemp, Btemp, Clocal, mloc, b);
        }
    }

	// Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start;

    if (IDENT) {
        bool local_success = true;
        bool global_success;

        for (int i = 0; i < mloc*nloc; i++)
            local_success = local_success && (Clocal[i] == Alocal[i]);

        MPI_Reduce(&local_success, &global_success, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
        if (!rank) assert(global_success);
    }

    // Print time
    if(!rank) {
        cout << nProcs << ","
			<< m << "," 
			<< p << "," 
			<< total_time << endl;
    }


    if (DEBUG) {
        ofstream logfile;
        stringstream s("");
        string filename;

        s << "log/" << ranki << ":" << rankj << ".log";
        s >> filename;

        logfile.open(filename.data());
        logfile << "Process (" << ranki << "," << rankj << ") [" << rank << "]:\n";
        logfile << "Alocal: \n";
        for (int i = 0; i < mloc; i++) {
            for (int j = 0; j < nloc; j++) {
                logfile << Alocal[i+j*mloc] << " ";
            }
            logfile << "\n";
        }
        logfile << "\nBlocal: \n";
        for (int i = 0; i < mloc; i++) {
            for (int j = 0; j < nloc; j++) {
                logfile << Blocal[i*nloc+j] << " ";
            }
            logfile << "\n";
        }

        logfile << "\nClocal: \n";
        for (int i = 0; i < mloc; i++) {
            for (int j = 0; j < nloc; j++) {
                logfile << Clocal[i+j*mloc] << " ";
            }
            logfile << "\n";
        }
        logfile << endl;
        logfile.close();
    }


    // Clean up
    delete [] Alocal;
    delete [] Blocal;
    delete [] Clocal;
    delete [] Atemp;
    delete [] Btemp;
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

void local_gemm(double* a, double* b, double* c, int m, int n) {
    // assume a is in column-major storage, and b is in row-major storage
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                // C(i,j) += A(i,k) * B(k,j)
                c[i+j*m] += a[i+k*m] * b[k*n+j];
            }
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
