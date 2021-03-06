#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <queue>
#include <vector>
#include <bits/stdc++.h> 

using namespace std;

void write_png(const char* filename, int iters, int width, int height, const int* buffer);
void color(int height);
void *working(void *data);
bool check_all_done();

int iters, n_thread, n_task_packet, current, task_size, _height, _width;
int *image;
int *work_done;
int done = 0;
double _lower, _upper, _left, _right;
pthread_mutex_t task_queue_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t work_done_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t task_queue_cond = PTHREAD_COND_INITIALIZER;

int BATCH_SIZE=500;

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    _left = strtod(argv[3], 0);
    _right = strtod(argv[4], 0);
    _lower = strtod(argv[5], 0);
    _upper = strtod(argv[6], 0);
    _width = strtol(argv[7], 0, 10);
    _height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(_width * _height * sizeof(int));
    assert(image);

    work_done = (int*)malloc(n_thread * sizeof(int));
    for(int i=0; i<n_thread; i++) {
        work_done[i] = 0;
    }
    current = 0;

    n_thread = CPU_COUNT(&cpu_set);
    pthread_t threads[n_thread];

    for(int i=0; i<n_thread; i++) {
        //cout << i << endl;
        pthread_create(&threads[i], NULL, working, new int(i));
        pthread_detach(threads[i]);
    }
    
    int start=0, end=0;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(current < _height) {
            start = current;
            current += 1;
        } else {
            start = current;
        }
        pthread_mutex_unlock(&task_queue_lock);
        //cout << start << endl;
        if(start < _height) {
            color(start);
        }

        if(start == _height) {
            break;
        }
    }

    while(!check_all_done()) {
        //cout << "wait" << endl;
    }

    /* draw and cleanup */
    write_png(filename, iters, _width, _height, image);
    free(image);
}

bool check_all_done() {
    // for(int i=0; i<n_thread; i++) {
    //     if(work_done[i] != 1) {
    //         //cout << "worker id: "<< i << endl;
    //         return false;
    //     }
    // }
    cout << done << endl;
    if(done != n_thread) return false;
    return true;
}

void *working(void *_id) {
    int *id = (int*) _id;
    int start=0, end=0;

    // cout << "worker id: "<< *id << endl;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(current < _height) {
            start = current;
            current += 1;
        } else {
            start = current;
        }
        pthread_mutex_unlock(&task_queue_lock);
        //cout << start << endl;
        if(start < _height) {
            color(start);
        }

        if(start == _height) {
            //work_done[*id] = 1;
            //cout << "worker id: "<< *id << endl;
            pthread_mutex_lock(&task_queue_lock);
            done += 1;
            pthread_mutex_unlock(&task_queue_lock);
            pthread_exit(NULL);
            break;
        }
    }
}

void color(int height) {
    int j = height;
    double y0 = j * ((_upper - _lower) / _height) + _lower;
    for (int i = 0; i < _width; ++i) {
        double x0 = i * ((_right - _left) / _width) + _left;

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
        image[j * _width + i] = repeats;
        //cout << repeats << endl;
    }
    
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
