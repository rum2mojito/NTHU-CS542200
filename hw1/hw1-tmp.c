#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

void _merge(float *tmp, float *a, float *b, int size_a, int size_b);
void _decompose(float *tmp, float *a, float *b, int size_a, int size_b);
void _sort(float *a, float *b, int size_a, int size_b);

void copy_array(float *a, float*b, int size_a);
void merge_low(float *a, float *b, int size_a, int size_b);
void merge_high(float *a, float *b, int size_a, int size_b);
int compare (void * a, void * b);

void print_buf(float *a, int size_a) {
	for (int i=0; i < size_a; i++) {
		printf("%f ", a[i]);
	}
	printf("\n");
}

void check(float* a, int size_a) {
	for (int i=0; i < size_a ; i++) {
		for (int j=i+1 ; j<size_a; j++) {
			if (a[i] > a[j]) {
				printf("Break: \n");
				print_buf(a, size_a);
				break;
			}
		}
	}
}

int is_odd(int a) {
	return a%2;
}

/*** main function ***/
int main(int argc, char** argv) {
	int rank, size, global_n, local_n, remainder, normal_n, sorted_result, right_n, left_n, sorted = 0, sum = 0;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_File f_in, f_out;

	int odd = is_odd(rank);
	global_n = atoll(argv[1]);
	remainder = global_n % size;
	normal_n = global_n /size;
	if (rank != 0) {
		local_n = normal_n;
		if (rank + 1 == size) {
			right_n = 0;
		} else {
			right_n = normal_n;
		}
		if (rank - 1 > 0) {
			left_n = normal_n;
		} else {
			left_n = normal_n + remainder;
		}
	} else {
		local_n = normal_n + remainder;
		right_n = normal_n;
		left_n = 0;
	}
	float *local_buf = malloc(local_n * sizeof(float));

	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f_in);
	if (rank != 0) {
		MPI_File_read_at(f_in, sizeof(float) * (rank*local_n+remainder), local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	} else {
		MPI_File_read_at(f_in, sizeof(float) * rank, local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&f_in);

	// merge_sort(local_buf, 0, local_n-1);
	qsort((float *)local_buf, local_n, sizeof(float), compare);

	float *recv_buf = malloc((normal_n + remainder) * sizeof(float));
	for (int i=0; i<size-1; i++) {
		MPI_Reduce(&sorted, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) {
			if (sum != size) {
				sorted = 0;
			}
		}
		MPI_Bcast(&sorted, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (sorted == 1) {
			break;
		} else {
			sorted = 1;
		}
		
		// even phase
		if (odd) {
			// BIG
			if (rank-1 >= 0) {
				MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank-1, 1, recv_buf, left_n, MPI_FLOAT, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (recv_buf[left_n-1] > local_buf[0]) {
					merge_high(recv_buf, local_buf, left_n, local_n);
					// sorted = 0;
				}
			}
		} else {
			// SMALL
			if (rank+1 < size) {
				MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank+1, 2, recv_buf, right_n, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (recv_buf[0] < local_buf[local_n-1]) {
					merge_low(local_buf, recv_buf, local_n, right_n);
					// sorted = 0;
				}
			}
		}
		// MPI_Barrier(MPI_COMM_WORLD);
		// odd pahse
		if (odd) {
			// SMALL
			if (rank+1 < size) {
				MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank+1, 1, recv_buf, right_n, MPI_FLOAT, rank+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				if (recv_buf[0] < local_buf[local_n-1]) {
					merge_low(local_buf, recv_buf, local_n, right_n);
					sorted = 0;
				}
			}
			
		} else {
			// BIG
			if (rank-1 >= 0) {
				MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank-1, 2, recv_buf, left_n, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				if (recv_buf[left_n-1] > local_buf[0]) {
					merge_high(recv_buf, local_buf, left_n, local_n);
					sorted = 0;
				}
			}
		}
	}

	// float *global_buf = malloc(global_n*sizeof(float));	
	// int *displs = (int *)malloc(size*sizeof(int));	
    // int *counts = (int *)malloc(size*sizeof(int));	
    // for (int i=0; i<size; ++i)  {	
	// 	if (i != 0) {	
    //     	displs[i] = normal_n*i + remainder;	
    //     	counts[i] = normal_n;	
	// 	} else {	
	// 		displs[i] = 0;	
    //     	counts[i] = local_n;	
	// 	}	
    // }
	
	// MPI_Gatherv(local_buf, local_n, MPI_FLOAT, global_buf, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	// if (rank == 0) {
	// 	printf("sorted %d\n  %f", global_n, global_buf[0]);
	// 	check(global_buf, global_n);
	// 	for(int i=0; i < global_n; i++) {	
	// 		for(int j=i+1; j<global_n;j++) {	
	// 			if(global_buf[j] < global_buf[i]) {	
	// 				printf("%d %f %d %f\n", j, global_buf[j], i, global_buf[i]);	
	// 			}	
	// 		}		
	// 	}
	// }

	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &f_out);
	if (rank != 0) {
		MPI_File_write_at(f_out, sizeof(float)*(rank*normal_n+remainder), local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	} else {
		MPI_File_write_at(f_out, 0, local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&f_out);
	
	free(local_buf);
	free(recv_buf);
	MPI_Finalize();
}

void copy_array(float *a, float*b, int size_a) {
	for (int i=0; i<size_a; i++) {
		b[i] = a[i];
	}
}

void merge_low(float *a, float *b, int size_a, int size_b) {
	int i_a = 0, i_b = 0, i = 0;
	float *tmp_a = malloc(size_a*sizeof(float));
	copy_array(a, tmp_a, size_a);

	while (i_a < size_a && i_b < size_b && i < size_a) {
		if (tmp_a[i_a] < b[i_b]) {
			a[i] = tmp_a[i_a];
			i++;
			i_a++;
		} else {
			a[i] = b[i_b];
			i++;
			i_b++;
		}
	}

	while (i_a < size_a && i < size_a) {
		a[i] = tmp_a[i_a];
		i++;
		i_a++;
	}

	while (i_b < size_b && i < size_a) {
		a[i] = b[i_b];
		i++;
		i_b++;
	}
	free(tmp_a);
	// check(a, size_a);
}

void merge_high(float *a, float *b, int size_a, int size_b) {
	int i_a = size_a-1, i_b = size_b-1, i = size_b-1;
	float *tmp_b = malloc(size_b*sizeof(float));
	copy_array(b, tmp_b, size_b);

	while (i_a >= 0 && i_b >= 0 && i >= 0) {
		if (a[i_a] > tmp_b[i_b]) {
			b[i] = a[i_a];
			i--;
			i_a--;
		} else {
			b[i] = tmp_b[i_b];
			i--;
			i_b--;
		}
	}

	while (i_a >= 0 && i >= 0) {
		b[i] = a[i_a];
		i--;
		i_a--;
	}

	while (i_b >= 0 && i >= 0) {
		b[i] = tmp_b[i_b];
		i--;
		i_b--;
	}
	free(tmp_b);
	// check(b, size_b);
}

int compare (void * a, void * b) {
  float fa = *(float*) a;
  float fb = *(float*) b;
  return (fa > fb) - (fa < fb);
}

/*** sort from different processes ***/
void _merge(float *tmp, float *a, float *b, int size_a, int size_b) {
	for (int i=0; i<(size_a+size_b); i++) {
		if (i < size_a) {
			tmp[i] = a[i];
		} else {
			tmp[i] = b[i-size_a];
		}
	}
}

void _decompose(float *tmp, float *a, float *b, int size_a, int size_b) {
	for (int i=0; i<(size_a+size_b); i++) {
		if (i < size_a) {
			a[i] = tmp[i];
		} else {
			b[i-size_a] = tmp[i];
		}
	}
}

void _sort(float *a, float *b, int size_a, int size_b) {
	float *tmp = malloc((size_a+size_b)*sizeof(float));
	_merge(tmp, a, b, size_a, size_b);
	qsort((float *)tmp, size_a+size_b, sizeof(float), compare);
	_decompose(tmp, a, b, size_a, size_b);
	
	free(tmp);
}
