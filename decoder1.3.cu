/*
LZ77 decoder
Parallel version 1.3.2

BSD 3-Clause License

Copyright (c) 2022, Kayla Wesley and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <sys/time.h>
#include <cub/device/device_scan.cuh>

#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "ERROR: could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(dst, src, size) if (cudaSuccess != cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of data to device failed\n");  CudaTest("data copy to device failed");
#define copyFromGPU(dst, src, size) if (cudaSuccess != cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of data from device failed\n");  CudaTest("data copy from device failed");

static void CudaTest(const char* msg)
{
  cudaError_t e;

  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}
typedef unsigned char byte;

static const int device = 0;
static const int offset = 256;
static const int ThreadsPerBlock = 512;

struct triple {
  byte dis; //distance to match
  byte len; //match length
  byte val; //next value
};

static __global__ void prefixSumPopulate(int* const __restrict__ prefix, const triple* const __restrict__ input, const int insize)
{
  // 2. Populate prefix sum array
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < insize) {
    prefix[idx] = (input[idx].dis == 0) ? 2 : (input[idx].len + 3);
  }
}

static __global__ void populateParentArray(int* const __restrict__ parent, const int* const __restrict__ prefix, const triple* const __restrict__ input, const int insize)
{
  // 4. Use prefix sum to populate parent array
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < insize) {
    const int start = (idx > 0) ? (prefix[idx - 1] + offset) : offset;
    if (input[idx].dis == 0) { // no matched values
      parent[start] = input[idx].val;
      parent[start + 1] = input[idx].len;
    } else { // matched values
      const int end = prefix[idx] + offset - 1;
      for (int j = start; j < end; j++) {
        parent[j] = j - input[idx].dis;
      }
      parent[end] = input[idx].val;
    }
  }
}

// Find operation
static inline __device__ int find(const int idx, volatile int* const __restrict__ parent)
{
  int curr = parent[idx];
  if (curr >= offset) {
    int prev = idx;
    do {
      const int next = parent[curr];
      parent[prev] = next;
      prev = curr;
      curr = next;
    } while (curr >= offset);
  }
  return curr;
}

static __global__ void populateOuput(byte* const __restrict__ output,  int* const __restrict__ parent, const int origLength)
{
  // 5. Populate output by union find
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < origLength) {
    output[idx] = find(idx + offset, parent);
  }
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char* argv[])
{
  printf("LZ77 (%s)\n", __FILE__);

  if (argc != 3) {printf("USAGE: %s input_file_name output_file_name\n", argv[0]);  exit(-1);}

  // 1. Read input
  FILE* const fin = fopen(argv[1], "rb");  assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  long size = ftell(fin);  assert(size > 0);
  long setSize = (size - sizeof(long)) / sizeof(triple);
  triple* const input = new triple [setSize];
  fseek(fin, 0, SEEK_SET);
  long origLength;
  long readElms = fread(&origLength, sizeof(long), 1, fin); assert(readElms == 1);
  if (origLength > INT_MAX) {printf("ERROR: the input file is too large for INT_MAX\n");  exit(-1);}
  const long insize = fread(input, (long)sizeof(triple), setSize, fin);  assert(insize == setSize);
  if (insize > INT_MAX) {printf("ERROR: the encoded file is too large for INT_MAX\n");  exit(-1);}
  fclose(fin);
  if (insize == 0) {printf("ERROR: input file is empty\n");  exit(-1);}

  // PLEASE NOTE: switching longs to int from here down....

  // Create prefix array, parent, & d_ variables
  int* d_prefix;
  int* d_parent;
  triple* d_input;

  // Allocate variables
  mallocOnGPU(d_prefix, sizeof(int) * insize);
  mallocOnGPU(d_input, sizeof(triple) * setSize);
  mallocOnGPU(d_parent, sizeof(int) * (origLength + offset));

  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  const int SMs = deviceProp.multiProcessorCount;
  printf("GPU: %s with %d SMs (%.1f MHz core and %.1f MHz mem)\n", deviceProp.name, SMs, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  
  // Create output array
  byte* const output = new byte [origLength];
  byte* d_output;
  mallocOnGPU(d_output, sizeof(byte) * origLength);  
  
  //Timer start_total
  timeval start, end, start_total, end_total;
  gettimeofday(&start_total, NULL);
  
  // Initialize variables
  copyToGPU(d_input, input, sizeof(triple) * setSize);

  // Determine temporary device storage requirements for inclusive prefix sum
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  
  gettimeofday(&start, NULL);
  // 2. Populate prefix sum array
  prefixSumPopulate<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_prefix, d_input, insize);
  // 3. Compute prefix sum array - https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
  // Determine temporary device storage requirements for inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prefix, d_prefix, insize);
  // Allocate temporary storage for inclusive prefix sum
  mallocOnGPU(d_temp_storage, temp_storage_bytes);
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prefix, d_prefix, insize);
  // 4. Use prefix sum to populate parent array
  populateParentArray<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_parent, d_prefix, d_input, insize);
  // 5. Populate output by union find
  populateOuput<<<(origLength + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_output, d_parent, origLength);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  printf("GPU runtime: %.6f s\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);

  // Get result from GPU
  //CheckCuda();
  copyFromGPU(output, d_output, sizeof(byte) * origLength);
  
  //Timer stop_total
  gettimeofday(&end_total, NULL);
  printf("Total runtime: %.6f s\n", end_total.tv_sec - start_total.tv_sec + (end_total.tv_usec - start_total.tv_usec) / 1000000.0);
  
  // 6. Write output
  FILE* const fout = fopen(argv[2], "wb");  assert(fout != NULL);
  size = fwrite(output, sizeof(byte), origLength, fout);  assert(size == origLength);
  fclose(fout);

  // Clean up
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_parent);
  cudaFree(d_prefix);
  cudaFree(d_temp_storage);
  cudaFreeHost(input);
  cudaFreeHost(output);
  return 0;
}