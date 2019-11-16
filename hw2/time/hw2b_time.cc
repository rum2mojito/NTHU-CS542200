#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <assert.h>
#include <png.h>
#include <string.h>
#include <omp.h>
#define PNG_NO_SETJMP
#define MAX_ITER 10000

using namespace std;
int num_threads;
struct timespec startt, endd;
struct timespec comp_time;

void calc_time(struct timespec* counter, struct timespec start, struct timespec endd) {
    struct timespec temp;
      if ((endd.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = endd.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+endd.tv_nsec-start.tv_nsec;
      } else {
        temp.tv_sec = endd.tv_sec-start.tv_sec;
        temp.tv_nsec = endd.tv_nsec-start.tv_nsec;
      }

    counter->tv_sec += temp.tv_sec;
    counter->tv_nsec += temp.tv_nsec;
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    num_threads = CPU_COUNT(&cpu_set);


    /*MPI setting*/
    int imagesize = width * height;
    int rank, psize, rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
	      printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort (MPI_COMM_WORLD, rc);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    comp_time.tv_sec = 0;
    comp_time.tv_nsec = 0;

    

    int h_each,h_piece;
    int remainder = 0;
    if(psize >= height){
      h_piece = 1;
    }
    else{
     h_piece = height/psize;
     h_each = h_piece;
     remainder = height%psize;
     if(remainder >0){
        if(rank < remainder){
            h_each++;

        }
      }
    }

    /* allocate memory for image */
    int* image;
    int* each_image = (int*)malloc(width * h_each * sizeof(int));

    clock_gettime(CLOCK_MONOTONIC, &startt);

    /* mandelbrot set */
    double y_dist = ((upper - lower) / height);
    double x_dist = ((right - left) / width);
    #pragma omp parallel num_threads(num_threads)
    {
      #pragma omp for schedule(dynamic)
      for (int h_index = 0; h_index < h_each; ++h_index) {
        int j = rank + psize*h_index;
        double y0 = j * y_dist + lower;
        int temp_index = h_index * width;
        for (int i = 0; i < width; ++i) {
            double x0 = i * x_dist + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            each_image[temp_index + i] = repeats;
        }
      }
    
    }
    
    image = (int*)malloc(imagesize * sizeof(int));
    int* revcount = (int*)malloc(psize*sizeof(int));
    int* displs = (int*)malloc(psize*sizeof(int));
    //memset(displs,0,psize*sizeof(int));
    displs[0] =0;
    for(int i=0;i<psize;++i){
      if(i<remainder){
        revcount[i] = (h_piece+1)*width;
        if(i+1<psize)displs[i+1] = displs[i] + revcount[i];
      }
      else {
        revcount[i] = h_piece*width;
        if(i+1<psize)displs[i+1] = displs[i] + revcount[i];
      }
    }
    MPI_Gatherv(each_image,h_each*width,MPI_INT,image,revcount,displs,MPI_INT,0,MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &endd);
    calc_time(&comp_time,startt,endd);
    double Comp_time_used = comp_time.tv_sec + (double)comp_time.tv_nsec / 1000000000.0;
    double allComp_time_used;
    MPI_Reduce(&Comp_time_used, &allComp_time_used, 1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank==0){
        /* index handle*/
        int* ans_image = (int*)malloc(imagesize * sizeof(int));
        int index = 0;
        for(int i=0;i<h_each;i++){
            int ri = 0;
            int jplus = h_each;
            for(int j=i;j<height && index < imagesize ;j+=jplus){
                int w0 = j*width;
                for(int w=0;w<width;w++){
                    ans_image[index] = image[w0 + w];
                    index++;
                }
                if(ri <remainder)jplus = h_each;
                else jplus = h_piece;
                ri++;
            }
        }
        /* draw and cleanup */
        write_png(filename, iters, width, height, ans_image);
        free(ans_image);

    }
    if(rank==0){
      if(psize <= height){
        allComp_time_used = allComp_time_used / psize;
      }
      else{
        allComp_time_used = allComp_time_used / height;
      }
      cout<<"Psize : " <<psize<<"\n";
      cout<<"Computing : "<<allComp_time_used<<"\n";
    }
    free(image);
    free(each_image);
    MPI_Finalize();
    return 0;
}
