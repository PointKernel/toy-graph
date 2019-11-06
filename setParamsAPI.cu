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
  kernel<<<1, 32, 0, stream>>>(dArray, size);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_h),
                  reinterpret_cast<void *>(dArray), size * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);

  cout << "Results from kernel:\n";
  print(A_h, size);

  // New data: host array
  float *B_h;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float)));
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));
  for (int i = 0; i < size; i++)
    B_h[i] = 1.f;
  cout << "Init new data:\n";
  print(B_h, size);

  size_t nNodes;
  // Get the number of graph nodes first
  cudaGraphGetNodes(graph, NULL, &nNodes);
  cout << "Number of nodes in graph: " << nNodes << "\n";
  // Then get the nodes
  cudaGraphNode_t nodes[nNodes];
  cudaGraphGetNodes(graph, nodes, &nNodes);

  // Get memcpy node parameters: cudaMemcpy3DParms
  // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemcpy3DParms.html#structcudaMemcpy3DParms
  cudaMemcpy3DParms dparams;
  cudaGraphMemcpyNodeGetParams(nodes[0], &dparams);

  // Print the memcpy kind
  // cudaMemcpyHostToHost = 0
  // cudaMemcpyHostToDevice = 1
  // cudaMemcpyDeviceToHost = 2
  // cudaMemcpyDeviceToDevice = 3
  // cudaMemcpyDefault = 4
  cout << "kind: " << dparams.kind << "\n";

  // Update nodes[0] host2device memcpy parameter
  cout << "\n######################## Update node 0\n";
  cout << "1st:  depth: " << dparams.extent.depth << "\t";
  cout << "height: " << dparams.extent.height << "\t";
  cout << "width: " << dparams.extent.width << "\n";

  dparams.srcPtr.ptr = B_h;
  dparams.dstPtr.ptr = dArray;
  dparams.extent.width = size * sizeof(float);
  cudaGraphMemcpyNodeSetParams(nodes[0], &dparams);
  cout << "B_h: " << B_h << "\t";
  cout << "srcPtr: " << dparams.srcPtr.ptr << "\n";

  cout << "dArray: " << dArray << "\t";
  cout << "dstPtr: " << dparams.dstPtr.ptr << "\n";

  cudaGraphMemcpyNodeGetParams(nodes[0], &dparams);
  cout << "2nd:  depth: " << dparams.extent.depth << "\t";
  cout << "height: " << dparams.extent.height << "\t";
  cout << "width: " << dparams.extent.width << "\n";

  // Update nodes[1] kernel parameter: cudaKernelNodeParams
  // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaKernelNodeParams.html#structcudaKernelNodeParams
  //
  // Kernel parameters can also be packaged by the application into a single
  // buffer that is passed in via the extra parameter. This places the burden on
  // the application of knowing each kernel parameter's size and
  // alignment/padding within the buffer. Here is an example of using the extra
  // parameter in this manner:
  //  void *config[] = {
  //    CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
  //    CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
  //    CU_LAUNCH_PARAM_END
  //  };
  cout << "\n######################## Update node 1\n";
  cudaKernelNodeParams kparams;
  cudaGraphKernelNodeGetParams(nodes[1], &kparams);
  cout << "1st:  blockDim.x: " << kparams.blockDim.x << "\t";
  cout << "blockDim.y: " << kparams.blockDim.y << "\t";
  cout << "blockDim.z: " << kparams.blockDim.z << "\n";
  kparams.blockDim.x = size;
  cout << "2nd:  blockDim.x: " << kparams.blockDim.x << "\t";
  cout << "blockDim.y: " << kparams.blockDim.y << "\t";
  cout << "blockDim.z: " << kparams.blockDim.z << "\n";

  cudaGraphKernelNodeSetParams(nodes[1], &kparams);

  // Update nodes[2] device2host memcpy parameter
  cout << "\n######################## Update node 2\n";
  cudaGraphMemcpyNodeGetParams(nodes[2], &dparams);

  dparams.srcPtr.ptr = dArray;
  dparams.dstPtr.ptr = B_h;
  dparams.extent.width = size * sizeof(float);
  cudaGraphMemcpyNodeSetParams(nodes[2], &dparams);
  cout << "B_h: " << B_h << "\t";
  cout << "dstPtr: " << dparams.dstPtr.ptr << "\n";

  cout << "dArray: " << dArray << "\t";
  cout << "srcPtr: " << dparams.srcPtr.ptr << "\n";

  cout << "Results from new data:\n";
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);
  print(B_h, size);

  cudaStreamDestroy(stream);

  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
  cudaFree(dArray);

  return 0;
}
