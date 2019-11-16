#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>

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

    fstream in_file, out_file;
    in_file.open(in_filename,ios::in | ios::binary);
    out_file.open(out_filename, ios::out | ios::binary);

    if(!in_file) {
        cout << "Open int_file fail." << endl;
    }
    if(!out_file) {
        cout << "Open int_file fail." << endl;
    }

    in_file.read((char *)&n_vertex, sizeof(n_vertex));
    in_file.read((char *)&n_edge, sizeof(n_edge));
    out_file.write((char*)(&n_vertex), sizeof(int));
    out_file.write((char*)(&n_edge), sizeof(int));
    cout << n_vertex << endl;
    cout << n_edge << endl;

    int src;
    int dst;
    int weight;
    for(int i=0;i<n_edge;i++){
        srand( time(NULL) );
        int x = rand();
        in_file.read((char *)&src, sizeof(src));
        in_file.read((char *)&dst, sizeof(dst));
        in_file.read((char *)&weight, sizeof(weight));
        weight += x%10;
        out_file.write((char*)(&src), sizeof(int));
        out_file.write((char*)(&dst), sizeof(int));
        out_file.write((char*)(&weight), sizeof(int));
        cout << src << endl;
        cout << dst << endl;
        cout << weight << endl;
    }
    in_file.close();
    out_file.close();   

    return 0;
}