#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <string>

using namespace std;

#define MAX 1073741823

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int n_threads = CPU_COUNT(&cpu_set);

    string in_filename;
    string out_filename;
    in_filename = argv[1];
    out_filename = argv[2];
    int n_vertex;
    int n_edge;

    fstream in_file;
    in_file.open(in_filename,ios::in | ios::binary);
    if(!in_file) {
        cout << "Open int_file fail." << endl;
    }
    in_file.read((char *)&n_vertex, sizeof(n_vertex));
    in_file.read((char *)&n_edge, sizeof(n_edge));

    // initializing graph
    int* dist = new int[n_vertex*n_vertex];
    #pragma omp parallel num_threads(n_threads) 
    {
        #pragma omp for collapse(2) schedule(static)
        for(int i=0; i<n_vertex; i++){
            for(int j=0; j<n_vertex; j++){
                if(i == j) {
                    dist[i*n_vertex + j] = 0;
                }
                else {
                    dist[i*n_vertex + j] = MAX;
                }
            }
        }
    }

    int src;
    int dst;
    int weight;
    for(int i=0;i<n_edge;i++){
        in_file.read((char *)&src, sizeof(src));
        in_file.read((char *)&dst, sizeof(dst));
        in_file.read((char *)&weight, sizeof(weight));
        dist[src*n_vertex + dst] = weight;
    }
    in_file.close();

    // Floyd-Warshall
    for(int i=0; i<n_vertex; i++) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp for collapse(2) schedule(static)
            for(int j=0; j<n_vertex; j++) {
                for(int k=0; k<n_vertex; k++) {
                    if(dist[j*n_vertex + k] > dist[j*n_vertex + i] + dist[i*n_vertex + k]) {
                        dist[j*n_vertex + k] = dist[j*n_vertex + i] + dist[i*n_vertex + k];
                    }
                }
            }
        }
    }

    fstream out_file;
    out_file.open(out_filename, ios::out | ios::binary);
    if(!out_file) {
        cout << "Open int_file fail." << endl;
    }
    for(int i=0; i<n_vertex; i++){
        for(int j=0; j<n_vertex; j++){
            out_file.write((char*)(dist + i*n_vertex + j), sizeof(int));
        }
    }
    out_file.close();

    return 0;
}