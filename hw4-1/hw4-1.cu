#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

void input(char* infile);
void output(char *outFileName);
int ceil(int a, int b);
void block_FW(int B);

int n, m;
int* Dist = NULL;
const int INF = ((1 << 30) - 1);
// const int V = 50010;

__global__ void cal(int *dist, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (dist[i*n + k] + dist[k*n + j] < dist[i*n + j]) {
                            dist[i*n + j] = dist[i*n + k] + dist[k*n + j];
                        }
                    }
                }
            }
        }
    }
    //__syncthreads();
}

int main(int argc, char* argv[]) {
	input(argv[1]);
	int B = 512;
	block_FW(B);
	output(argv[2]);
    cudaFreeHost(Dist);

	return 0;
}

void block_FW(int B) {
	int round = ceil(n, B);
    int* dis = NULL;
	cudaSetDevice(0);
	//size_t pitch;
    cudaMalloc(&dis,sizeof(int)*n*n);
    //cudaMallocPitch(&dis,&pitch,(size_t)sizeof(int)*n,(size_t)n);
	cudaMemcpy(dis,Dist,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    //cudaMemcpy2D(dis, pitch, Dist,(size_t)sizeof(int)*n, (size_t)sizeof(int)*n,(size_t)n,cudaMemcpyHostToDevice);
    
    //dim3 num_threads(128, 4);
    const int num_blocks = 1;
    const int num_threads = 1024;

	//APSP
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal<<<num_blocks, num_threads>>>(dis, B, r, r, r, 1, 1, n);

        /* Phase 2*/
        cal<<<num_blocks, num_threads>>>(dis, B, r, r, 0, r, 1, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, r, r + 1, round - r - 1, 1, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, 0, r, 1, r, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, r + 1, r, 1, round - r - 1, n);

        /* Phase 3*/
        cal<<<num_blocks, num_threads>>>(dis, B, r, 0, 0, r, r, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, 0, r + 1, round - r - 1, r, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, r + 1, 0, r, round - r - 1, n);
        cal<<<num_blocks, num_threads>>>(dis, B, r, r + 1, r + 1, round - r - 1, round - r - 1, n);
    }

	cudaMemcpy(Dist,dis,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
	//cudaMemcpy2D(Dist,(size_t)sizeof(int)*n,dis,pitch,(size_t)sizeof(int)*n,(size_t)n,cudaMemcpyDeviceToHost);
	cudaFree(dis);
}

void input(char* infile) {
    cout << "input" << endl;
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (int*) malloc(sizeof(int)*n*n);
    //cudaMallocHost((void**) &Dist, sizeof(int) * n*n);

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = pair[2];
        //cout << "("<<pair[0]<<','<<pair[1]<<")"<<pair[2]<<'\n';
    }
    fclose(file);

    // for (int i = 0; i < n; ++ i) {
    //     for (int j = 0; j < n; ++ j) {
    //         cout << Dist[i*n+j] << '\t';
    //     }
    //     cout << endl << endl;
    // }
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (Dist[i*n+j] >= INF)
                Dist[i*n+j] = INF;
                cout << Dist[i*n+j] << '\t';
            //else cout << "("<<i<<','<<j<<")"<<Dist[i*n+j]<<'\n';
        }
        cout << endl;
		fwrite(&Dist[i*n], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }
