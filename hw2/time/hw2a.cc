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

void write_png(const char* filename, int iters, int width, int height, const int* buffer);
int color(double x0, double y0, int iters);
void *working(void *data);
bool check_all_done();

struct task{
    int index;
    double x0;
    double y0;

    task(int index1, double x01, double y01) {
        index = index1;
        x0 = x01;
        y0 = y01;
    }
};

queue<task*> task_queue;
vector<task*> task_list;
int iters, n_thread, n_task_packet, current, task_size;
int *image;
int *work_done;
pthread_mutex_t task_queue_lock = PTHREAD_MUTEX_INITIALIZER;
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
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    comp_time.tv_sec = 0;
    comp_time.tv_nsec = 0;

    clock_gettime(CLOCK_MONOTONIC, &startt);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    work_done = (int*)malloc(n_thread * sizeof(int));
    for(int i=0; i<n_thread; i++) {
        work_done[i] = 0;
    }
    current = 0;

    n_thread = CPU_COUNT(&cpu_set);
    pthread_t threads[n_thread];

    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;
            int index = j * width + i;
            task *new_task = new task(index, x0, y0);
            task_list.push_back(new_task);
        }
    }
    shuffle(task_list.begin(), task_list.end(), default_random_engine(0));
    task_size = task_list.size();


    for(int i=0; i<n_thread; i++) {
        pthread_create(&threads[i], NULL, working, new int(i));
        pthread_detach(threads[i]);
    }

    task *new_task1 = NULL;
    int start=0, end=0;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(current < task_size) {
            if(task_size-current > BATCH_SIZE) {
                start = current;
                end = current+BATCH_SIZE;
                current += BATCH_SIZE;
            } else {
                start = current;
                end = task_size;
                current = task_size;
            }
        }
        pthread_mutex_unlock(&task_queue_lock);

        for(int i=start; i<end; i++) {
            new_task1 = task_list[i];
            if(new_task1 != NULL) {
                image[new_task1->index] = color(new_task1->x0, new_task1->y0, iters);
                //delete new_task;
                new_task1 = NULL;
            }
        }

        if(current == task_size) {
            break;
        }
    }
    while(!check_all_done()) {
        //cout << "wait" << endl;
    }

    clock_gettime(CLOCK_MONOTONIC, &endd);
    calc_time(&comp_time, startt, endd);
    double Comp_time_used = comp_time.tv_sec + (double)comp_time.tv_nsec / 1000000000.0;
    cout<<"Computing : "<<Comp_time_used<<"\n";

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

bool check_all_done() {
    for(int i=0; i<n_thread; i++) {
        if(work_done[i] != 1) return false;
    }
    return true;
}

void *working(void *_id) {
    int *id = (int*) _id;
    //cout << *id << endl;
    task *new_task = NULL;
    int start=0, end=0;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(current < task_size) {
            if(task_size-current > BATCH_SIZE) {
                start = current;
                end = current+BATCH_SIZE;
                current += BATCH_SIZE;
            } else {
                start = current;
                end = task_size;
                current = task_size;
            }
        }
        pthread_mutex_unlock(&task_queue_lock);

        for(int i=start; i<end; i++) {
            new_task = task_list[i];
            if(new_task != NULL) {
                image[new_task->index] = color(new_task->x0, new_task->y0, iters);
                //delete new_task;
                new_task = NULL;
            }
        }

        if(current == task_size) {
            work_done[*id] = 1;
            pthread_exit(NULL);
            break;
        }
    }
}

int color(double x0, double y0, int iters) {
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

    return repeats;
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
