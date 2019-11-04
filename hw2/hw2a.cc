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

using namespace std;

void write_png(const char* filename, int iters, int width, int height, const int* buffer);
int color(double x0, double y0, int iters);
void *working(void *data);

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
int iters, n_thread;
int *image;
pthread_mutex_t task_queue_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t task_queue_cond = PTHREAD_COND_INITIALIZER;

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

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    n_thread = CPU_COUNT(&cpu_set);
    pthread_t threads[n_thread];

    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;
            int index = j * width + i;
            task *new_task = new task(index, x0, y0);
            task_queue.push(new_task);
        }
    }


    for(int i=0; i<n_thread; i++) {
        pthread_create(&threads[i], NULL, working, NULL);
        pthread_detach(threads[i]);
    }

    task *new_task = NULL;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(!task_queue.empty()) {
            new_task = task_queue.front();
            task_queue.pop();
        }
        pthread_mutex_unlock(&task_queue_lock);
        if(new_task != NULL) {
            image[new_task->index] = color(new_task->x0, new_task->y0, iters);
            delete new_task;
            new_task = NULL;
        }
        if(task_queue.size() == 0) {
            cout << task_queue.size() << endl;
            break;
        }
    }
    //working(NULL);
    // for(int i=0; i<n_thread; i++) {
    //     pthread_join(threads[i], NULL);
    // }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

void master_working() {
    task *new_task = NULL;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(!task_queue.empty()) {
            new_task = task_queue.front();
            task_queue.pop();
        }
        pthread_mutex_unlock(&task_queue_lock);
        if(new_task != NULL) {
            image[new_task->index] = color(new_task->x0, new_task->y0, iters);
            delete new_task;
            new_task = NULL;
        }
        if(task_queue.size() == 0) {
            cout << task_queue.size() << endl;
            break;
        }
    }
}

void *working(void *data) {
    task *new_task = NULL;
    
    while(1) {
        //cout<< "busy" << endl;
        pthread_mutex_lock(&task_queue_lock);
        if(!task_queue.empty()) {
            new_task = task_queue.front();
            task_queue.pop();
        }
        pthread_mutex_unlock(&task_queue_lock);
        if(new_task != NULL) {
            image[new_task->index] = color(new_task->x0, new_task->y0, iters);
            delete new_task;
            new_task = NULL;
        }
        if(task_queue.size() == 0) {
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
