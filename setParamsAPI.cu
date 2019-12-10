#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>

using namespace std;

const int SIZE = 8;

void init(float *A, int size) {
  for (int i = 0; i < size; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A, int size) {
  for (int i = 0; i < size; i++) {
    cout << A[i] << " ";
  }
  cout << "\n\n";
}

__global__ void kernel(float *array, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    array[index] += 1.f;
    if (index == 0)
      printf("### Array size: %d\n", size);
  }
}

int main() {
  int size = SIZE;

  dim3 nblocks(1, 1, 1);
  dim3 nthreads(size, 1, 1);

  // Host array
  float *A_h;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&A_h), size * sizeof(float)));

  // Device array
  float *dArray;
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));

  init(A_h, size);
  cout << "Results after init:\n";
  print(A_h, size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Create graph by capturing
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(dArray),
                  reinterpret_cast<void *>(A_h), size * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  kernel<<<nblocks, nthreads, 0, stream>>>(dArray, size);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_h),
                  reinterpret_cast<void *>(dArray), size * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  // Run graph for the first time
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);

  cout << "Results from kernel:\n";
  print(A_h, size);

  // Change array elements
  for (int i = 0; i < size; i++)
    A_h[i] = 1.f;
  cout << "Init new data:\n";
  print(A_h, size);
  
  // Update graph instance
  cudaGraphExecUpdateResult resUpdate;
  checkCudaErrors(cudaGraphExecUpdate(instance, graph, NULL, &resUpdate));

  // Relaunch the graph with updated parameters
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  cout << "\nResults from new data:\n";
  print(A_h, size);

  cudaStreamDestroy(stream);

  cudaFreeHost(A_h);
  cudaFree(dArray);

  return 0;
}
