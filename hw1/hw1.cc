#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <bits/stdc++.h> 

using namespace std;

void copy_array(float *a, float*b, int size_a);
void merge_low(float *a, float *b, int size_a, int size_b);
void merge_high(float *a, float *b, int size_a, int size_b);
int is_odd(int a);

void print_buf(float *a, int size_a) {
	for (int i=0; i < size_a; i++) {
		printf("%f ", a[i]);
	}
	printf("\n");
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
	float *local_buf = new float[(local_n)];

	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f_in);
	if (rank != 0) {
		MPI_File_read_at(f_in, sizeof(float) * (rank*local_n+remainder), local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	} else {
		MPI_File_read_at(f_in, sizeof(float) * rank, local_buf, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&f_in);

	sort(local_buf, local_buf+local_n);

	float *recv_buf = new float[((normal_n + remainder))];
	float *recv_check_point, in = 0;
	recv_check_point = &in;
	for (int i=0; i<size; i++) {
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
				MPI_Sendrecv(&local_buf[0], 1, MPI_FLOAT, rank-1, 3, recv_check_point, 1, MPI_FLOAT, rank-1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (*recv_check_point > local_buf[0]) {
					MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank-1, 1, recv_buf, left_n, MPI_FLOAT, rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					merge_high(recv_buf, local_buf, left_n, local_n);
				}
			}
		} else {
			// SMALL
			if (rank+1 < size) {
				MPI_Sendrecv(&local_buf[local_n-1], 1, MPI_FLOAT, rank+1, 4, recv_check_point, 1, MPI_FLOAT, rank+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (*recv_check_point < local_buf[local_n-1]) {
					MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank+1, 2, recv_buf, right_n, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					merge_low(local_buf, recv_buf, local_n, right_n);
				}
			}
		}
		// MPI_Barrier(MPI_COMM_WORLD);
		// odd pahse
		if (odd) {
			// SMALL
			if (rank+1 < size) {
				MPI_Sendrecv(&local_buf[local_n-1], 1, MPI_FLOAT, rank+1, 3, recv_check_point, 1, MPI_FLOAT, rank+1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (*recv_check_point < local_buf[local_n-1]) {
					//cout << *recv_check_point << " " << local_buf[local_n-1] << endl;
					MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank+1, 1, recv_buf, right_n, MPI_FLOAT, rank+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					merge_low(local_buf, recv_buf, local_n, right_n);
					sorted = 0;
				}
			}
			
		} else {
			// BIG
			if (rank-1 >= 0) {
				MPI_Sendrecv(&local_buf[0], 1, MPI_FLOAT, rank-1, 4, recv_check_point, 1, MPI_FLOAT, rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				if (*recv_check_point > local_buf[0]) {
					MPI_Sendrecv(local_buf, local_n, MPI_FLOAT, rank-1, 2, recv_buf, left_n, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					merge_high(recv_buf, local_buf, left_n, local_n);
					sorted = 0;
				}
			}
		}
	}

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
	float *tmp_a = new float[(size_a)];
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
}

void merge_high(float *a, float *b, int size_a, int size_b) {
	int i_a = size_a-1, i_b = size_b-1, i = size_b-1;
	float *tmp_b = new float[(size_b)];
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
}

int is_odd(int a) {
	return a%2;
}
