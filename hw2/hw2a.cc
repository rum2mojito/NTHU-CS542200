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
//void *job(void *data1);

#include <iostream>
#include <queue>
#include <pthread.h>
#include <unistd.h>
 
using namespace std;

int iters = 0;
int n_thread = 0;
int count = 0;
int* image;

 
class job {
public:
    job(int index, double x0, double y0) {
        this->index = index;
        this->x0 = x0;
        this->y0 = y0;
    };
    virtual ~job(){}
    //will be overrided by specific JOB
    void virtual working()
    {
        pthread_mutex_lock(&jobLock);
        finished_jobs++;
        pthread_mutex_unlock(&jobLock);
        //cout << "JOB:" << index << " starts!\n";
        int repets = color(x0, y0, iters);
        //cout << "repeat: " << repets << endl;
        image[index] = repets;
    }
    static int finished_jobs;
    static pthread_mutex_t jobLock;
private:
    int index;
    double x0;
    double y0;
};
 
class thread_pool {
public:
    static pthread_mutex_t jobQueue_lock;
    static pthread_cond_t jobQueue_cond;
    thread_pool(){ thread_pool(2); }
    thread_pool(int num) : numOfThreads(num) {}
    virtual ~thread_pool() { while(!jobQueue.empty()) jobQueue.pop(); };
    void initThreads(pthread_t *);
    void assignJob(job *_job_);
    bool loadJob(job *&_job_);
    static void *threadExecute(void *);
private:
    queue<job*> jobQueue;
    int numOfThreads;
};
 
int job::finished_jobs = 0;
 
pthread_mutex_t job::jobLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t thread_pool::jobQueue_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t thread_pool::jobQueue_cond = PTHREAD_COND_INITIALIZER;
 
void thread_pool::initThreads(pthread_t *threads)
{
     
    for(int i = 0; i < numOfThreads; i++)
    {
        pthread_create(&threads[i], NULL, &thread_pool::threadExecute, (void *)this);
        cout << "Thread:" << i << " is alive now!\n";
    }
}
 
void thread_pool::assignJob(job* _job_)
{
    pthread_mutex_lock(&jobQueue_lock);
    jobQueue.push(_job_);
    pthread_mutex_unlock(&jobQueue_lock);
    pthread_cond_signal(&jobQueue_cond);
}
 
bool thread_pool::loadJob(job*& _job_)
{
    pthread_mutex_lock(&jobQueue_lock);
    while(jobQueue.empty())
        pthread_cond_wait(&jobQueue_cond, &jobQueue_lock);
    _job_ = jobQueue.front();
    jobQueue.pop();
    pthread_mutex_unlock(&jobQueue_lock);
    return true;
}
 
void *thread_pool::threadExecute(void *param)
{
    thread_pool *p = (thread_pool *)param;
    job *oneJob = NULL;
    while(p->loadJob(oneJob))
    {
        if(oneJob)
            oneJob->working();
        delete oneJob;
        oneJob = NULL;
    }
}


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
    thread_pool *myPool = new thread_pool(n_thread);
    pthread_t threads[n_thread];
    myPool->initThreads(threads);

    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;
            int index = j * width + i;
            job *newJob = new job(index, x0, y0);
            myPool->assignJob(newJob);
            // image[j * width + i] = color(x0, y0, iters);
        }
    }
    while(job::finished_jobs < height*width) {
        cout << job::finished_jobs << endl;
    }
    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
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
