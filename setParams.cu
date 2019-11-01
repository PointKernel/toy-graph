#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>

const int SIZE = 32;

void init(float *A) {
  for (size_t i = 0; i < SIZE; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A) {
  for (size_t i = 0; i < SIZE; i++) {
    std::cout << A[i] << " ";
  }
  std::cout << "\n";
}

__global__ void kernel(float *array) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < SIZE)
    array[index] += 1.f;
}

int main() {
  // Host array
  float *A_h;
  checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&A_h), SIZE * sizeof(float)));

  // Device array
  float *A_d;
  cudaMalloc(reinterpret_cast<void **>(&A_d), SIZE * sizeof(float));

  init(A_h);
  std::cout << "results after init:\n";
  print(A_h);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Create graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h),
                  SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
  kernel<<<1, 32, 0, stream>>>(A_d);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_h), reinterpret_cast<void *>(A_d),
                  SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));

  std::cout << "results from kernel:\n";
  print(A_h);

  // size_t numP, numQ;
  // cudaGraphGetNodes(graphP, NULL, &numP);
  // cudaGraphGetNodes(graphQ, NULL, &numQ);

  // cudaGraphNode_t nodesP[numP], nodesQ[numQ];
  // cudaGraphGetNodes(graphP, nodesP, &numP);
  // cudaGraphGetNodes(graphQ, nodesQ, &numQ);

  // cudaKernelNodeParams knp;
  // cudaGraphKernelNodeGetParams(nodesQ[0], &knp);
  // cudaGraphExecKernelNodeSetParams(graphExec, nodesP[0], &knp);
  // cudaGraphLaunch(graphExec, stream);
  // cudaDeviceSynchronize();

  cudaStreamDestroy(stream);

  cudaFreeHost(A_h);
  cudaFree(A_d);

  return 0;
}
