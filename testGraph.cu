#include <cstdlib>
#include <iostream>

using namespace std;

const int CONST = 128;

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
  for (int i = index; i < n; i += stride) {
    if (x[i] > y[i]) {
      for (int j = 0; j < n/CONST; j++)
        y[i] = x[j] + y[j];
    }
    else {
      for (int j = 0; j < n/CONST; j++)
        y[i] = x[j] / y[j];
    }
  }
}

__global__ void kernelB(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    if (x[i] > y[i]) {
      for (int j = 0; j < n/CONST; j++)
        y[i] = x[j] + y[j];
    }
    else {
      y[i] = atomicAdd(&y[i], x[i]);
    }
  }
}

__global__ void kernelC(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    if (x[i] > y[i]) {
      for (int j = 0; j < n/CONST; j++)
        y[i] = x[j] + y[j];
    }
}

__global__ void kernelD(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    for (int j = 0; j < n/CONST; j++)
      y[i] = atomicAdd(&y[j], x[j]);
  }
}

int main(int argc, char *argv[]) {
  int size;
  if (argc == 2) {
    cout << "\nArray size: " << argv[1] << endl;
    size = atoi(argv[1]);
  } else {
    size = 1 << 16;
    cout << "\nUsing default matrix size: " << size << endl;
  }

  const int nStreams = 4;

  // One cudaGraphExec_t per stream is required
  cudaGraph_t graph;
  cudaGraphExec_t instance[nStreams];

  // Declare host data
  float *A_h[nStreams];
  float *B_h[nStreams];
  float *C_h[nStreams];

  for (int i = 0; i < nStreams; i++) {
    cudaMallocHost(reinterpret_cast<void **>(&A_h[i]), size * sizeof(float));
    cudaMallocHost(reinterpret_cast<void **>(&B_h[i]), size * sizeof(float));
    cudaMallocHost(reinterpret_cast<void **>(&C_h[i]), size * sizeof(float));
  }

  // Declare device data
  float *A_d[nStreams];
  float *B_d[nStreams];
  float *C_d[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaMalloc(reinterpret_cast<void **>(&A_d[i]), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&B_d[i]), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&C_d[i]), size * sizeof(float));
  }

  // Initialize host data
  for (int i = 0; i < nStreams; i++)
    init(size, A_h[i], B_h[i], C_h[i]);

  // Create CUDA events for timing measurement
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

  // Create graph once
  cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeGlobal);  // begin of the graph
  cudaMemcpyAsync(reinterpret_cast<void *>(A_d[0]), reinterpret_cast<void *>(A_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(reinterpret_cast<void *>(B_d[0]), reinterpret_cast<void *>(B_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);
  kernelA<<<gridDim, blockDim, 0, stream[0]>>>(size, A_d[0], B_d[0]);

  cudaMemcpyAsync(reinterpret_cast<void *>(C_d[0]), reinterpret_cast<void *>(C_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);
  kernelB<<<gridDim, blockDim, 0, stream[0]>>>(size, B_d[0], C_d[0]);

  kernelC<<<gridDim, blockDim, 0, stream[0]>>>(size, C_d[0], A_d[0]);
  cudaMemcpyAsync(reinterpret_cast<void *>(C_d[0]), reinterpret_cast<void *>(C_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);

  kernelD<<<gridDim, blockDim, 0, stream[0]>>>(size, A_d[0], B_d[0]);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_d[0]), reinterpret_cast<void *>(A_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(reinterpret_cast<void *>(B_d[0]), reinterpret_cast<void *>(B_h[0]), size,
                  cudaMemcpyHostToDevice, stream[0]);
  cudaStreamEndCapture(stream[0], &graph); // end of the graph
  // create an instance per stream
  for(int i=0; i < nStreams; ++i) {
    cudaGraphInstantiate(&instance[i], graph, NULL, NULL, 0);
  }

  for (size_t i = 0; i < 1000; i++) {
    int idStream = i % nStreams;

    // How to use A_d[idStream], A_h[idStream], B_d[idStream], B_h[idStream], C_d[idStream], C_h[idstream]?
    // As of now the kernels and transfers are full of data races...

    // Launch graph
    cudaGraphLaunch(instance[idStream], stream[idStream]);
  }

  cudaEventRecord(stop);

  // Print total runtime
  cudaEventSynchronize(stop);
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = static_cast<double>(milliseconds) / 1000.;
  cout << "runtime: " << seconds << endl;

  // Print to prevent the compiler from over optimization
  for (size_t i = 0; i < nStreams; i++) {
    cout << A_h[i][CONST] << endl;
    cout << B_h[i][CONST] << endl;
    cout << C_h[i][CONST] << endl;
  }
  
  for (size_t i = 0; i < nStreams; i++)
    cudaStreamDestroy(stream[i]);

  // Free the allocated memory
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
