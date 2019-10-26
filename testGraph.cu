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
  uint64_t size;
  if (argc == 2) {
    cout << "\nArray size: " << argv[1] << endl;
    size = atoi(argv[1]);
  } else {
    size = 1 << 24;
    cout << "\nUsing default matrix size: " << size << endl;
  }

  // declare host data
  float *A_h;
  float *B_h;
  float *C_h;
  cudaMallocHost(reinterpret_cast<void **>(&A_h), size * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&C_h), size * sizeof(float));

  // declare device data
  float *A_d;
  float *B_d;
  float *C_d;
  cudaMalloc(reinterpret_cast<void **>(&A_d), size * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&B_d), size * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&C_d), size * sizeof(float));

  // initialize host data
  init(size, A_h, B_h, C_h);

  // create CUDA events for timing measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // dim3 gridDim;
  // dim3 blockDim;
  const uint64_t gridDim = 1024;
  const uint64_t blockDim = 64u;

  const int nStreams = 4;
  cudaStream_t stream[nStreams];

  for (size_t i = 0; i < nStreams; i++)
    cudaStreamCreate(&stream[i]);
  
  cudaMemcpy(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h), size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(reinterpret_cast<void *>(B_d), reinterpret_cast<void *>(B_h), size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(reinterpret_cast<void *>(C_d), reinterpret_cast<void *>(C_h), size,
             cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  for (size_t i = 0; i < 1000; i++) {
    int idStream = i % nStreams;
    // copy host data to device
    cudaMemcpyAsync(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h), 1024,
               cudaMemcpyHostToDevice, stream[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_d), reinterpret_cast<void *>(B_h), 1024,
               cudaMemcpyHostToDevice, stream[idStream]);

    kernelA<<<gridDim, blockDim, 0, stream[idStream]>>>(size, A_d, B_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(C_d), reinterpret_cast<void *>(C_h), 512,
               cudaMemcpyHostToDevice, stream[idStream]);

    kernelB<<<gridDim, blockDim, 0, stream[idStream]>>>(size, B_d, C_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_d), reinterpret_cast<void *>(B_h), 1024,
               cudaMemcpyHostToDevice, stream[idStream]);
    cudaMemcpyAsync(reinterpret_cast<void *>(C_d), reinterpret_cast<void *>(C_h), 512,
               cudaMemcpyHostToDevice, stream[idStream]);

    kernelC<<<gridDim, blockDim, 0, stream[idStream]>>>(size, C_d, A_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(C_d), reinterpret_cast<void *>(C_h), 512,
               cudaMemcpyHostToDevice, stream[idStream]);
    kernelD<<<gridDim, blockDim, 0, stream[idStream]>>>(size, A_d, B_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h), 1024,
               cudaMemcpyHostToDevice, stream[idStream]);
  }

  cudaEventRecord(stop);

  cudaMemcpy(reinterpret_cast<void *>(A_h), reinterpret_cast<void *>(A_d), size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(reinterpret_cast<void *>(C_h), reinterpret_cast<void *>(C_d), size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(reinterpret_cast<void *>(B_h), reinterpret_cast<void *>(B_d), size,
             cudaMemcpyDeviceToHost);

  // print kernel runtime
  cudaEventSynchronize(stop);
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = static_cast<double>(milliseconds) / 1000.;
  cout << "runtime: " << seconds << endl;
  
  for (size_t i = 0; i < 4; i++)
    cudaStreamDestroy(stream[i]);

  // free the allocated memory
  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
  cudaFreeHost(C_h);
  cudaFree(reinterpret_cast<void *>(A_d));
  cudaFree(reinterpret_cast<void *>(B_d));
  cudaFree(reinterpret_cast<void *>(C_d));

  return 0;
}
