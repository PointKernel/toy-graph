#include <cstdlib>
#include <iostream>

using namespace std;

void init(uint64_t size, float *A, float *B, float *C) {
  for (size_t i = 0; i < size; i++) {
    A[i] = static_cast<float>(rand() % 100);
    B[i] = static_cast<float>(rand() % 100);
    C[i] = static_cast<float>(rand() % 100);
  }
}

__global__ void kernelA(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__ void kernelB(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * 3.14f;
}

__global__ void kernelC(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = logf(atanf(x[i]) / cosf(expf(x[i])));
}

__global__ void kernelD(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = log(x[i] / expf(y[i]));
}

int main(int argc, char *argv[]) {
  int size;
  if (argc == 2) {
    cout << "\nArray size: " << argv[1] << endl;
    size = atoi(argv[1]);
  } else {
    size = 1 << 24;
    cout << "\nUsing default matrix size: " << size << endl;
  }

  const int nStreams = 4;

  // declare host data
  float *A_h[nStreams];
  float *B_h[nStreams];
  float *C_h[nStreams];

  for (int i = 0; i < nStreams; i++) {
    cudaMallocHost(reinterpret_cast<void **>(&A_h[i]), size * sizeof(float));
    cudaMallocHost(reinterpret_cast<void **>(&B_h[i]), size * sizeof(float));
    cudaMallocHost(reinterpret_cast<void **>(&C_h[i]), size * sizeof(float));
  }

  // declare device data
  float *A_d[nStreams];
  float *B_d[nStreams];
  float *C_d[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaMalloc(reinterpret_cast<void **>(&A_d[i]), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&B_d[i]), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&C_d[i]), size * sizeof(float));
  }

  // initialize host data
  for (int i = 0; i < nStreams; i++)
    init(size, A_h[i], B_h[i], C_h[i]);

  // create CUDA events for timing measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // dim3 gridDim;
  // dim3 blockDim;
  const int gridDim = 1024;
  const int blockDim = 64;

  cudaStream_t stream[nStreams];

  for (size_t i = 0; i < nStreams; i++)
    cudaStreamCreate(&stream[i]);
  
  cudaEventRecord(start);

  for (size_t i = 0; i < 1000; i++) {
    int idStream = i % nStreams;
    // copy host data to device
    cudaMemcpyAsync(reinterpret_cast<void *>(A_d[idStream]), reinterpret_cast<void *>(A_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_d[idStream]), reinterpret_cast<void *>(B_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);
    kernelA<<<gridDim, blockDim, 0, stream[idStream]>>>(size, A_d[idStream], B_d[idStream]);

    cudaMemcpyAsync(reinterpret_cast<void *>(C_d[idStream]), reinterpret_cast<void *>(C_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);
    kernelB<<<gridDim, blockDim, 0, stream[idStream]>>>(size, B_d[idStream], C_d[idStream]);

    kernelC<<<gridDim, blockDim, 0, stream[idStream]>>>(size, C_d[idStream], A_d[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(C_d[idStream]), reinterpret_cast<void *>(C_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);

    kernelD<<<gridDim, blockDim, 0, stream[idStream]>>>(size, A_d[idStream], B_d[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(A_d[idStream]), reinterpret_cast<void *>(A_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_d[idStream]), reinterpret_cast<void *>(B_h[idStream]), size,
               cudaMemcpyHostToDevice, stream[idStream]);
  }

  cudaEventRecord(stop);

  // print kernel runtime
  cudaEventSynchronize(stop);
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = static_cast<double>(milliseconds) / 1000.;
  cout << "runtime: " << seconds << endl;
  
  for (size_t i = 0; i < nStreams; i++)
    cudaStreamDestroy(stream[i]);

  // free the allocated memory
  for (size_t i = 0; i < nStreams; i++) {
    cudaFreeHost(A_h[i]);
    cudaFreeHost(B_h[i]);
    cudaFreeHost(C_h[i]);
    cudaFree(reinterpret_cast<void *>(A_d[i]));
    cudaFree(reinterpret_cast<void *>(B_d[i]));
    cudaFree(reinterpret_cast<void *>(C_d[i]));
  }

  return 0;
}
