#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N 3500      // Matrix size
#define ITMAX 1000  // Iterations

double (*A)[N];
double (*B)[N];

int main(int argc, char **argv)
{
    MPI_Request request[4];
    MPI_Status status[4];
    
    int rank, numtasks;
    int startrow, lastrow, nrow;
    
    double starttime, endtime;
    double total_sum = 0.;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Barrier(MPI_COMM_WORLD);

    startrow = (rank * N) / numtasks;
    lastrow = (((rank + 1) * N) / numtasks) - 1;
    nrow = lastrow - startrow + 1;
    
    if (rank == 0) {
        printf("PROGRAMM STARTED\n");
        printf("Number of processes: %d\n", numtasks);
    }
    // Initialize matrices
    A = malloc((nrow + 2) * N * sizeof(double));
    B = malloc((nrow + 2) * N * sizeof(double));

    for (int i = 0; i < nrow + 2; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 0.;
            if ((rank == 0 && i == 0) || (rank == numtasks - 1 && i == nrow + 1)
                || j == 0 || j == N - 1) {
                B[i][j] = 0.;
            } else {
                B[i][j] = 1. + i + j + startrow;
            }
        }
    }

    // Start timer for each process
    starttime = MPI_Wtime();
    for (int it = 0; it < ITMAX; ++it) {

        // Copy matrix
        for (int i = 1; i <= nrow; ++i) {   
            if ((i == 1 && rank == 0) || (i == nrow && rank == numtasks - 1)) {
                continue;
            }
            for (int j = 1; j < N - 1; ++j) {
                A[i][j] = B[i][j];
            }
        }
      
        // Send and receive shadow edges
        if (rank != 0) {
            MPI_Irecv(&A[0][0], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &request[0]);
        }
        if (rank != numtasks - 1) {
            MPI_Isend(&A[nrow][0], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &request[2]);
        }
        if (rank != numtasks - 1) {
            MPI_Irecv(&A[nrow + 1][0], N, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &request[3]);
        }
        if (rank != 0) {
            MPI_Isend(&A[1][0], N, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &request[1]);
        }

        int wait_for = 4;
        int shift = 0;
        
        if (rank == 0) {
            wait_for = 2;
            shift = 2;
        }

        if (rank == numtasks - 1) {
            wait_for = 2;
        }

        if (numtasks == 1) {
            wait_for = 0; // only one process, no need to wait
            shift = 0;
        }

        // Wait for all exchanges
        MPI_Waitall(wait_for, &request[shift], &status[0]);

        // Perform operations
        for (int i = 1; i <= nrow; ++i) {
            if ((i == 1 && rank == 0) || (i == nrow && rank == numtasks - 1)) {
                continue;
            }
            for (int j = 1; j < N - 1; ++j) {
                B[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
            }
        }
    }

    // Stop timer
    endtime = MPI_Wtime();
    printf("%d: Time of the process = %lf\n", rank, endtime - starttime);

    // Calculate local sum
    double local_sum = 0.;
    for (int i = 1; i <= nrow; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            local_sum += A[i][j] * (i + startrow) * j / (N * N);
        }
    }

    // Calculate total sum
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();

    if (rank == 0) {
        printf("S = %lf\n", total_sum);
    }

    return 0;
}
