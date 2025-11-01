/*
LZ77 coder modified for easier parallelization
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
#include <utility>
#include <chrono>  // Changed from sys/time.h

// Windows compatibility
#ifdef _WIN32
#include <windows.h>
#endif

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
static const int MAXLEN = 256;
static const int ThreadsPerBlock = 512;

struct triple {
  byte dis; //distance to match
  byte len; //match length
  byte val; //next value
};

static __global__ void findMatches(byte* const __restrict__ matchdis, byte* const __restrict__ matchlen, const byte* const __restrict__ input, const long insize)
{
  __shared__ byte buf [512 + ThreadsPerBlock];

  //1. Find matches in the 256 window
  const int first = blockIdx.x * ThreadsPerBlock;
  const int idx = threadIdx.x + first;
  if (idx + 256 < insize) {
    buf[512 + threadIdx.x] = input[idx + 256];
  }
  if (idx - 256 >= 0) {
    buf[threadIdx.x] = input[idx - 256];
  }
  __syncthreads();

  if (idx < insize) {
    long maxlen = 0;
    long maxidx;
    long pos = idx - 1;
    while ((idx - pos < MAXLEN) && (pos >= 0)) {
      if (buf[256 - first + pos] == buf[256 + threadIdx.x]) {
        long len = 1;
        //Find longest match
        while ((len <= MAXLEN) && (idx + len < insize) && (buf[256 - first + pos + len] == buf[256 + threadIdx.x + len])) {
          len++;
        }
        if (maxlen <= len) { //will save the largest or farthest within the window
          maxlen = len;
          maxidx = pos;
        }
      }
      pos--;
    }
    if (maxlen < 2) {
      matchdis[idx] = 0;
    } else {
      matchdis[idx] = idx - maxidx;
      matchlen[idx] = maxlen - 2;
    }
  }
}

static inline __device__ int find(int prev, volatile int* const __restrict__ rep)
{
  int next, curr = rep[prev];
  if (curr != prev) {
    while (curr > (next = rep[curr])) {
      rep[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

static inline __device__ void combine(int r1, int r2, volatile int* const __restrict__ rep)  // union
{
  int ma;
  do {
    if (r1 == r2) break;
    ma = max(r1, r2);
    r2 = min(r1, r2);
  } while ((r1 = atomicCAS((int*)&rep[ma], ma, r2)) != ma);
}

static __global__ void init(const int insize, short int* const __restrict__ len, int* const __restrict__ rep, int* const __restrict__ reach, byte* const __restrict__ matchdis, byte* const __restrict__ matchlen)
{
  const int i = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (i < insize) {
    len[i] = (matchdis[i] == 0) ? 2 : (int)matchlen[i] + 2 + 1;
    rep[i] = i;
    reach[i] = (i) ? 0 : 1;
  }
}

static __global__ void reachable(const int insize, const short int* const __restrict__ len, int* const __restrict__ reach)
{
  // count reachability
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    const int pos = i + len[i];
    if (pos < insize) {
      atomicAdd(&reach[pos], 1);
    }
  }
}

static __global__ void chains(const int insize, const short int* const __restrict__ len, volatile int* const __restrict__ rep, const int* const __restrict__ reach)
{
  // build chains (union)
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    const int pos = i + len[i];
    if ((pos < insize) && (reach[pos] == 1)) {
      rep[pos] = i;
    }
  }
}

static __global__ void undangle(const int insize, const short int* const __restrict__ len, volatile int* const __restrict__ rep, volatile int* const __restrict__ reach)
{
  // remove dangling chains
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    const int pos = i + len[i];
    if ((pos < insize) && (reach[pos] > 1)) {
      const int r = find(i, rep);
      if (reach[r] == 0) {
        reach[r] = -1;
        atomicSub((int*)&reach[pos], 1);
      }
    }
  }
}

static __global__ void merge(const int insize, volatile bool* const __restrict__ goagain, const short int* const __restrict__ len, volatile int* const __restrict__ rep, const int* const __restrict__ reach)
{
  // merge linked lists (union) at non-fork points
  bool go = false;
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    const int pos = i + len[i];
    if ((pos < insize) && (reach[pos] == 1)) {
      int r1 = find(i, rep);
      if (reach[r1] >= 0) {
        int r2 = find(pos, rep);
        if (r1 != r2) {
          go = true;
          combine(r1, r2, rep);
        }
      }
    }
  }
  if (__syncthreads_or(go)) {
    if (threadIdx.x == 0) *goagain = true;
  }
}

static __global__ void check(const int insize, int* const __restrict__ count, volatile int* const __restrict__ rep)
{
  // check count (remove later)
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    if (find(i, rep) == 0) atomicAdd((int*)count, 1);
  }
}

static __global__ void initpfs(const int insize, volatile int* const __restrict__ rep, byte* const __restrict__ locations)
{
  // init prefix sum
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < insize) {
    locations[i] = (find(i, rep) == 0) ? 1 : 0;
  }
}

static __global__ void populatePrefixArray(int* const __restrict__ prefix, byte* const __restrict__ locations, const long insize)
{
  //3.a populate prefix array
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < insize) {
    prefix[idx] = locations[idx];
  }
}

static __global__ void makeTriples(byte* const __restrict__ matchdis, byte* const __restrict__ matchlen, int* const __restrict__ prefix, byte* const __restrict__ locations, byte* const __restrict__ input, triple* output, const long insize)
{
  //4. make triples for the output array.
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < insize) {
    if (locations[idx] != 0) {
      long index = idx;
      long outindex = (idx == 0) ? 0 : (prefix[idx - 1]);
      if (matchdis[idx] == 0) {
        output[outindex].dis = 0;
        output[outindex].val = (index < insize) ? input[index] : 0;
        index++;
        output[outindex].len = (index < insize) ? input[index] : 0;
      } else {
        output[outindex].dis = matchdis[index];
        output[outindex].len = matchlen[index];
        index += (int)matchlen[index] + 2;
        output[outindex].val = (index < insize) ? input[index] : 0;
      }
    }
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
  printf("LZ77 encoder (%s)\n", __FILE__);
  if (argc != 3) {printf("USAGE: %s input_file_name output_file_name\n", argv[0]);  exit(-1);}

  //memory check
  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  printf("     %.1f GB/s peak bandwidth (%d-bit bus)\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 0.000001, deviceProp.memoryBusWidth);

  // Read input
  FILE* const fin = fopen(argv[1], "rb");  assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  long size = ftell(fin);  assert(size > 0);
  byte* input;  // = new byte [size];
  cudaMallocHost(&input, size * sizeof(byte));
  fseek(fin, 0, SEEK_SET);
  const long insize = fread(input, sizeof(byte), size, fin);  assert(insize == size);
  fclose(fin);
  if (insize == 0) {printf("ERROR: input file is empty\n");  exit(-1);}

  // Create output
  triple* output;  // = new triple [insize];  // upper bound
  cudaMallocHost(&output, insize * sizeof(triple));
  // Device variables
  int* d_prefix;
  byte* d_input;
  byte* d_matchdis;
  byte* d_matchlen;
  byte* d_locations;
  triple* d_output;
  short int* d_len;
  int* d_reach;
  int* d_rep;
  // Determine temporary device storage requirements for inclusive prefix sum
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Allocate variables
  mallocOnGPU(d_input, sizeof(byte) * insize);
  mallocOnGPU(d_matchdis, sizeof(byte) * insize);
  mallocOnGPU(d_matchlen, sizeof(byte) * insize);
  mallocOnGPU(d_output, sizeof(triple) * insize);
  mallocOnGPU(d_len, sizeof(short int) * insize);
  mallocOnGPU(d_reach, sizeof(int) * insize);
  mallocOnGPU(d_rep, sizeof(int) * insize);

  // Timer start_total - Using chrono for cross-platform timing
  auto start_total = std::chrono::high_resolution_clock::now();
  
  // Initialize variables
  copyToGPU(d_input, input, sizeof(byte) * insize);

  // Timer for Kernels
  auto start = std::chrono::high_resolution_clock::now();

  // 1. Find matches
  findMatches<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_matchdis, d_matchlen, d_input, insize);
  // 2. Mark location of matches

  init<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_len, d_rep, d_reach, d_matchdis, d_matchlen);
  reachable<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_len, d_reach);
  //CudaTest("reachable");
  chains<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_len, d_rep, d_reach);
  //CudaTest("chains");

  int c = 0;
  bool goagain;
  bool* d_goagain;
  mallocOnGPU(d_goagain, sizeof(bool));
  do {
    c++;
    goagain = false;
    copyToGPU(d_goagain, &goagain, sizeof(bool));
    undangle<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_len, d_rep, d_reach);
    //CudaTest("undangle");
    merge<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_goagain, d_len, d_rep, d_reach);
    //CudaTest("merge");
    copyFromGPU(&goagain, d_goagain, sizeof(bool));
  } while (goagain);
  cudaFree(d_goagain);
  printf("%d iterations\n", c);
  cudaFree(d_len);
  cudaFree(d_reach);
  mallocOnGPU(d_locations, sizeof(byte) * insize);
  initpfs<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(insize, d_rep, d_locations);
  //CudaTest("initpfs");

  cudaFree(d_rep);
  mallocOnGPU(d_prefix, sizeof(int) * insize);
  // 3. Populate prefix array
  populatePrefixArray<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_prefix, d_locations, insize);
  // 4. Compute prefix sum array - https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
  // 4.a Determine temporary device storage requirements for inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prefix, d_prefix, insize);
  // 4.b Allocate temporary storage for inclusive prefix sum
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // 4.c Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prefix, d_prefix, insize);
  // 5. Create & Store triples in output array
  int outsize = 0;
  copyFromGPU(&outsize, d_prefix + insize - 1, sizeof(int));
  CheckCuda();
  // 6. Make and store triples
  makeTriples<<<(insize + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_matchdis, d_matchlen, d_prefix, d_locations, d_input, d_output, insize);

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("GPU runtime:   %.6f s\n", elapsed.count());

  //CheckCuda();
  copyFromGPU(output, d_output, sizeof(triple) * outsize);

  // Timer stop_total
  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_total = end_total - start_total;
  printf("Total runtime: %.6f s\n", elapsed_total.count());

  // Write output
  FILE* const fout = fopen(argv[2], "wb");  assert(fout != NULL);
  size = fwrite(&insize, sizeof(long), 1, fout);  assert(1 == size);
  size = fwrite(output, sizeof(triple), outsize, fout);  assert(outsize == size);
  fclose(fout);

  // Print compression ratio
  printf("Compression ratio: %.3f\n", 1.0 * insize / (sizeof(long) + sizeof(triple) * outsize));

  // Clean up
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_prefix);
  cudaFree(d_locations);
  cudaFree(d_matchdis);
  cudaFree(d_matchlen);
  cudaFree(d_temp_storage);
  cudaFreeHost(input);
  cudaFreeHost(output);
  return 0;
}