#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>

using namespace std;

const int SIZE = 8;

void init(float *A) {
  for (size_t i = 0; i < SIZE; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A) {
  for (size_t i = 0; i < SIZE; i++) {
    cout << A[i] << " ";
  }
  cout << "\n\n";
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
  cout << "Results after init:\n";
  print(A_h);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Create graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h),
                  SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
  kernel<<<1, SIZE, 0, stream>>>(A_d);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_h), reinterpret_cast<void *>(A_d),
                  SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));

  cout << "Results from kernel:\n";
  print(A_h);

  size_t nNodes;
  // Get the number of graph nodes first
  cudaGraphGetNodes(graph, NULL, &nNodes);
  cout << "Number of nodes in graph: " << nNodes << "\n";
  // Then get the nodes
  cudaGraphNode_t nodes[nNodes];
  cudaGraphGetNodes(graph, nodes, &nNodes);

  // Get memcpy node parameters
  // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemcpy3DParms.html#structcudaMemcpy3DParms
  cudaMemcpy3DParms params;
  cudaGraphMemcpyNodeGetParams(nodes[0], &params);

  // Print the memcpy kind
  // cudaMemcpyHostToHost = 0
  // cudaMemcpyHostToDevice = 1
  // cudaMemcpyDeviceToHost = 2
  // cudaMemcpyDeviceToDevice = 3
  // cudaMemcpyDefault = 4
  cout << "kind: " << params.kind << "\n";

  cout << "depth: " << params.extent.depth << "\t";
  cout << "height: " << params.extent.height << "\t";
  cout << "width: " << params.extent.width << "\n";
  
  cout << "A_h: " << A_h << "\t";
  cout << params.srcPtr.ptr << "\n";

  cout << "A_d: " << A_d << "\t";
  cout << params.dstPtr.ptr << "\n";

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
