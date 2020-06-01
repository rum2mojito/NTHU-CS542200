#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

const int INF = 1000000000;

int n, m;
int* Dist = NULL;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    /* hw4 */
    //Dist = (int*) malloc(sizeof(int)*n*n);
    cudaMallocHost((void**) &Dist, sizeof(int) * n*n);

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

int ceil(int a, int b) {
	return (a + b - 1) / b;
}

__global__ void APSP_phase1(int* dis,int B, int Round,int num,size_t pitch)
{
        // since maxthreadperblock = 1024 , max B = 32
        // phase1 only 1 block
        extern __shared__ int shared_dis[];

        int x = threadIdx.y,//x = threadIdx.x,
            y = threadIdx.x,//y = threadIdx.y,
            orx = Round * B + x, // original index x
            ory = Round * B + y; // original index y

           //shared_dis[x*B +y] = ( orx < num && ory < num) ? dis[orx*num + ory] : INF;
           shared_dis[x*B +y] = ( orx < num && ory < num) ? ((int*)((char*)dis+orx*pitch))[ory] : INF;
           __syncthreads();

        #pragma unroll
        for (int k = 0; k < B; ++k) {
            int temp = shared_dis[x*B+k] + shared_dis[k*B+y];
            if(shared_dis[x*B+y] > temp) shared_dis[x*B+y] = temp;
            __syncthreads();
        }
        //if(orx < num && ory < num)dis[orx*num + ory] = shared_dis[x*B + y];
        if(orx < num && ory < num)((int*)((char*)dis+orx*pitch))[ory] = shared_dis[x*B + y];
        __syncthreads();

}
__global__ void APSP_phase2(int* dis,int B, int Round,int num,size_t pitch)
{
        if(blockIdx.x == Round )return; //don't need to cal pivot again

        extern __shared__ int shared_memory[];
        int* pivot = &shared_memory[0]; // store pivot value
        int* shared_dis = &shared_memory[B*B]; // ans of this block
        int x = threadIdx.y,//x = threadIdx.x,
            y = threadIdx.x,//y = threadIdx.y,
            orx = Round * B + x, // original index x of pivot
            ory = Round * B + y; // original index y of pivot

        //pivot[x*B + y] = ( orx < num && ory < num)? dis[orx*num + ory] : INF;
        pivot[x*B + y] = ( orx < num && ory < num)? ((int*)((char*)dis+orx*pitch))[ory] : INF;

        if(blockIdx.y == 0 )ory = blockIdx.x*B + y; //row
        else orx = blockIdx.x*B +x; //column

        if (orx >= num || ory >= num) return;
        //shared_dis[x*B + y] = (orx < num && ory < num)? dis[orx*num + ory] : INF;
        shared_dis[x*B + y] = (orx < num && ory < num)? ((int*)((char*)dis+orx*pitch))[ory] : INF;
        __syncthreads();

        if (blockIdx.y == 1) {
            #pragma unroll
            for (int k = 0; k < B; ++k) {
                int temp = shared_dis[x*B + k] + pivot[k*B + y];
                if (shared_dis[x*B + y] > temp) shared_dis[x*B + y] = temp;
            }
        }
        else {
            #pragma unroll
            for (int k = 0; k < B; ++k) {
                int temp = pivot[x*B + k] + shared_dis[k*B + y];
                if (shared_dis[x*B + y] > temp) shared_dis[x*B + y] = temp;
            }
        }

     //if(orx < num && ory < num) dis[orx*num + ory] = shared_dis[x*B + y];
     if(orx < num && ory < num) ((int*)((char*)dis+orx*pitch))[ory] = shared_dis[x*B + y];

}
__global__ void APSP_phase3(int* dis,int B, int Round,int num,size_t pitch)
{
    if(blockIdx.x == Round || blockIdx.y == Round)return; // just need to cal other blocks

    extern __shared__ int shared_memory[];
    int* shared_row = &shared_memory[0];
    int* shared_cloumn = &shared_memory[B*B];
    int x = threadIdx.y,//x = threadIdx.x,
        y = threadIdx.x,//y = threadIdx.y,
        orx = Round * B + x, // original index x of pivot
        ory = Round * B + y, // original index y of pivot
        i = blockIdx.x * blockDim.x + x, // original index x of cal block
        j = blockIdx.y * blockDim.y + y; // original index y of cal block

    //shared_row[x*B + y] = (i < num && ory < num)? dis[i*num + ory] : INF;
    //shared_cloumn[x*B + y] = (orx < num && j < num )? dis[orx*num + j] : INF;
    shared_row[x*B + y] = (i < num && ory < num)? ((int*)((char*)dis+i*pitch))[ory] : INF;
    shared_cloumn[x*B + y] = (orx < num && j < num )?((int*)((char*)dis+orx*pitch))[j] : INF;
     __syncthreads();

    if(i >= num || j >= num)return;

    //int d = dis[i*num + j];
    int d = ((int*)((char*)dis+i*pitch))[j];
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int temp = shared_row[x*B + k] + shared_cloumn[k*B + y];
        if (d > temp)d = temp;
    }
    //dis[i*num + j] = d;
    ((int*)((char*)dis+i*pitch))[j] = d;


}

void block_FW(int B) {
	int round = ceil(n, B);
  int* dis = NULL;
	cudaSetDevice(0);
	size_t pitch;
    //cudaMalloc(&dis,sizeof(int)*n*n);
    cudaMallocPitch(&dis,&pitch,(size_t)sizeof(int)*n,(size_t)n);
	//cudaMemcpy(dis,Dist,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy2D(dis,pitch,Dist,(size_t)sizeof(int)*n,(size_t)sizeof(int)*n,(size_t)n,cudaMemcpyHostToDevice);
    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3(round, round);
    dim3 threads(B, B);

	// time tracking
    cudaEvent_t start, stop;
	  cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//APSP
	cudaEventRecord(start, 0);
	for (int r = 0; r < round; ++r) {
        //printf("%d %d\n", r, round);
        //fflush(stdout);
		/* Phase 1*/
    APSP_phase1<<<grid_phase1,threads,B*B*sizeof(int)>>>(dis, B, r, n,pitch);
		/* Phase 2*/
		APSP_phase2<<<grid_phase2,threads,B*B*sizeof(int)*2>>>(dis, B, r, n,pitch);

		/* Phase 3*/
		APSP_phase3<<<grid_phase3,threads,B*B*sizeof(int)*2>>>(dis, B, r, n,pitch);
	}

	// time tracking
	  cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start, stop);
    //printf("cuda_time = %lf\n",cuda_time);
    //fflush(stdout);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	//cudaMemcpy(Dist,dis,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy2D(Dist,(size_t)sizeof(int)*n,dis,pitch,(size_t)sizeof(int)*n,(size_t)n,cudaMemcpyDeviceToHost);
	cudaFree(dis);
}

int main(int argc, char* argv[]) {
	input(argv[1]);
	//int B = 512;
	int B;
	if(argc >=4)B = atoi(argv[3]);
	else B = 30;
	if(B > n)B = n;
	block_FW(B);
	output(argv[2]);
    //free(Dist);
     cudaFreeHost(Dist);

	return 0;
}
