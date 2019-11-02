#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <omp.h>
int main(int argc, char *argv[]) {
    int rank, size;
    long long numParts;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    numParts = atoll(argv[1]);

    double sum;
    double msg = 0;
    double x = numParts;
#pragma omp parallel for reduction( +:msg)
    for(int i=rank; i < numParts; i+=size) {
        msg += sqrt(1 - pow(i/x, 2));
    }
    // MPI_Send(&msg, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    MPI_Reduce(&msg, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("%.10f\n", sum/x*4);
    }

    MPI_Finalize();
    return 0;
}
