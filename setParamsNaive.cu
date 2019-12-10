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
      printf("### array[%d] = %f\tArray size: %d\n", index, array[index], size);
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

  // Create graph
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
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);

  cout << "Results from kernel:\n";
  print(A_h, size);
  
  size *= 2;
  dim3 nthds(size, 1, 1);

  float *B_h;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float)));

  for (int i = 0; i < size; i++)
    B_h[i] = 5.f;
  cout << "Init new data:\n";
  print(B_h, size);

  cudaMemcpy3DParms dParams0, dParams2;
  cudaKernelNodeParams kParams;

  size_t nNodes;
  // Get the number of graph nodes first
  cudaGraphGetNodes(graph, NULL, &nNodes);
  cout << "Number of nodes in graph: " << nNodes << "\n";
  // Then get nodes
  cudaGraphNode_t nodes[nNodes];
  cudaGraphGetNodes(graph, nodes, &nNodes);

  // #################### Update node 0 ####################
  cudaGraphMemcpyNodeGetParams(nodes[0], &dParams0);
  //dParams0.srcArray = NULL;
  //dParams0.srcPos = make_cudaPos(0, 0, 0);
  dParams0.srcPtr = make_cudaPitchedPtr(B_h, size * sizeof(float), 1, 1);
  //dParams0.dstArray = NULL;
  //dParams0.dstPos = make_cudaPos(0, 0, 0);
  dParams0.dstPtr = make_cudaPitchedPtr(dArray, size * sizeof(float), 1, 1);
  dParams0.extent = make_cudaExtent(size * sizeof(float), 1, 1);
  //dParams0.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaGraphExecMemcpyNodeSetParams(instance, nodes[0], &dParams0));

  // #################### Update node 1 ####################
  cudaGraphKernelNodeGetParams(nodes[1], &kParams);
  //kParams.func = (void *)kernel;
  //kParams.gridDim = nblocks;
  kParams.blockDim = nthds;
  //kParams.sharedMemBytes = 0;
  void *kernelArgs[2] = {(void *)&dArray, (void *)&size};
  kParams.kernelParams = kernelArgs;
  //kParams.extra = NULL;
  checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, nodes[1], &kParams));

  // #################### Update node 2 ####################
  cudaGraphMemcpyNodeGetParams(nodes[2], &dParams2);
  //dParams2.srcArray = NULL;
  //dParams2.srcPos = make_cudaPos(0, 0, 0);
  dParams2.srcPtr = make_cudaPitchedPtr(dArray, size * sizeof(float), 1, 1);
  //dParams2.dstArray = NULL;
  //dParams2.dstPos = make_cudaPos(0, 0, 0);
  dParams2.dstPtr = make_cudaPitchedPtr(B_h, size * sizeof(float), 1, 1);
  dParams2.extent = make_cudaExtent(size * sizeof(float), 1, 1);
  //dParams2.kind = cudaMemcpyDeviceToHost;
  checkCudaErrors(cudaGraphExecMemcpyNodeSetParams(instance, nodes[2], &dParams2));

  // Relaunch the graph with new parameters
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  cout << "\nResults from new data:\n";
  print(B_h, size);

  cudaStreamDestroy(stream);

  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
  cudaFree(dArray);

  return 0;
}
