#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

struct thread_data {
  long long numParts;
  int id;
  long long normal_n;
  long long remainder;
  double result;
};

void* cal(void *data1) {
  struct thread_data *data = (struct thread_data*) data1;
  printf("%d %d\n", data->normal_n, data->remainder);
  data->result = 0.0;
  if (data->id == 0) {
    for (long long i = 0; i < data->normal_n+data->remainder; i++) {
      data->result += sqrt(1 - ((double)i / data->numParts) * ((double)i / data->numParts));
    }
  } else {
    for (long long i = data->normal_n*data->id + data->remainder; i < data->normal_n*(data->id+1) + data->remainder; i++) {
      data->result += sqrt(1 - ((double)i / data->numParts) * ((double)i / data->numParts));
    }
  }
  printf("id %d: %f\n", data->id, data->result);
}

int main(int argc, char **argv) {
  double pi = 0;
  double res = 0;

  long long numParts = atoll(argv[2]);

  int num_threads = atoi(argv[1]);
  pthread_t threads[num_threads];
  int rc;
  int ID[num_threads];
  int result[num_threads];
  long long normal_n = numParts/num_threads;
  long long remainder = numParts % num_threads;
  struct thread_data *data = malloc(sizeof(struct thread_data) * num_threads);

  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  for(int i=0; i<num_threads; i++) {
    data[i].id = i;
    data[i].normal_n = normal_n;
    data[i].numParts = numParts;
    data[i].remainder = remainder;
    // printf("%d\n", numParts);
    rc = pthread_create(&threads[i], NULL, cal, &data[i]);
  }
  
  for (int i=0; i<num_threads; i++) {
    pthread_join(threads[i], NULL);
    // while(pthread_mutex_trylock(&mutex) != 0);
    // pthread_mutex_lock(&mutex);
    res += data[i].result;
    // pthread_mutex_unlock(&mutex);
  }

  printf("%.12lf\n", res * 4.0 / numParts);

  return 0;
}
