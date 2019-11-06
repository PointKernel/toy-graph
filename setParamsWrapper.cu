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

__global__ void kernel(float **array, int *size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < *size) {
    *array[index] = 2.f;
    if (index == 0)
      printf("### Array size: %d\n", *size);
  }
}

template <typename T> struct MemCpy {
  int *size;
  T **pHstPtr;
  T **pDevPtr;
  MemCpy(){};
  MemCpy(int *s, T **h, T **d) : size(s), pHstPtr(h), pDevPtr(d) {}
};

int main() {
  int size = SIZE;

  // Host array
  float *A_h;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&A_h), size * sizeof(float)));

  // Device array
  float *dArray;
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));

  auto memcpy = MemCpy<float>(&size, &A_h, &dArray);

  init(A_h, size);
  cout << "Results after init:\n";
  print(*memcpy.pHstPtr, size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Create graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(*memcpy.pDevPtr),
                  reinterpret_cast<void *>(*memcpy.pHstPtr),
                  *memcpy.size * sizeof(float), cudaMemcpyHostToDevice, stream);
  kernel<<<1, 32, 0, stream>>>(memcpy.pDevPtr, memcpy.size);
  cudaMemcpyAsync(reinterpret_cast<void *>(*memcpy.pHstPtr),
                  reinterpret_cast<void *>(*memcpy.pDevPtr),
                  *memcpy.size * sizeof(float), cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);
  /*
    cout << "Results from kernel:\n";
    print((float *)*memcpy.hstPtr, size);

    // New data: host array
    float *B_h;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&B_h), size *
    sizeof(float))); cudaMalloc(reinterpret_cast<void **>(&dArray), size *
    sizeof(float)); memcpy.size = &size; memcpy.hstPtr = (void **) &B_h;
    memcpy.devPtr = (void **) &dArray;
    for (int i = 0; i < size; i++)
      B_h[i] = 1.f;
    cout << "Init new data:\n";
    print((float *)*memcpy.hstPtr, *memcpy.size);

    checkCudaErrors(cudaGraphLaunch(instance, stream));
    cudaStreamSynchronize(stream);
    cout << "Results from new data:\n";
    print((float *)*memcpy.hstPtr, *memcpy.size);

    cudaStreamDestroy(stream);

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFree(dArray);
  */
  return 0;
}
