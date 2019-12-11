#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int size = 1;
const int N = 512 * 512;

void init(float *A) {
  for (int i = 0; i < size; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A) {
  for (int i = 0; i < size; i++) {
    cout << A[i] << " ";
  }
  cout << "\n\n";
}

__global__ void kernel(float *array) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  array[index] += 1.f;
}

void runGraph(cudaGraphExec_t& instance, cudaStream_t& stream) {
  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    // Relaunch the graph with new parameters
    checkCudaErrors(cudaGraphLaunch(instance, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time = duration_cast<duration<double>>(end - start);
  cout << "Graph runtime:    " << time.count() << "\n";
}

void runKernels(float* B_h, float* B_d, int nKernels, cudaStream_t &stream) {
  dim3 nblocks(1);
  dim3 nthreads(size);

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(reinterpret_cast<void *>(B_d),
                    reinterpret_cast<void *>(B_h), size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    for(int j = 0; j < nKernels; j++)
      kernel<<<nblocks, nthreads, 0, stream>>>(B_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_h),
                    reinterpret_cast<void *>(B_d), size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time = duration_cast<duration<double>>(end - start);
  cout << "Kernels runtime:  " << time.count() << "\n";
}

int main() {
  float *B_h;
  float *B_d;

  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&B_d), size * sizeof(float)));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  dim3 nblocks(1);
  dim3 nthreads(size);

  for (int nKernels = 1; nKernels <= 20; nKernels++) {
    cout << "\n##### Kernel number: " << nKernels << "\tMemcpy number: " << 2 << " #####\n";

  
    init(B_h);
    //cout << "Init:\n";
    //print(B_h);

    // Create graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(reinterpret_cast<void *>(B_d),
                    reinterpret_cast<void *>(B_h), size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < nKernels; i++)
      kernel<<<nblocks, nthreads, 0, stream>>>(B_d);
    cudaMemcpyAsync(reinterpret_cast<void *>(B_h),
                    reinterpret_cast<void *>(B_d), size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamEndCapture(stream, &graph);

    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    checkCudaErrors(cudaGraphLaunch(instance, stream));
    cudaStreamSynchronize(stream);

    //cout << "First run (init value + " << nKernels << "):\n";
    //print(B_h);

    // Run graph
    init(B_h);
    runGraph(instance, stream);
    //print(B_h);

    // Run kernels
    init(B_h);
    runKernels(B_h, B_d, nKernels, stream);
    //print(B_h);
  }

  cudaStreamDestroy(stream);

  cudaFreeHost(B_h);
  cudaFree(B_d);

  return 0;
}
