/*
LZ77 decoder simplified for easier parallelization
Serial version 1.2.1

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
#include <chrono>  // Changed from sys/time.h

typedef unsigned char byte;

static const long offset = 256;

struct triple {
  byte dis; //distance to match
  byte len; //match length
  byte val; //next value
};

// Find operation
static inline long find(const long idx, long* const parent)
{
  long curr = parent[idx];
  if (parent[curr] >= offset) {
    long next, prev = idx;
    while (parent[curr] >= offset) {
      next = parent[curr];
      parent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
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
  const long insize = fread(input, (long)sizeof(triple), setSize, fin);  assert(insize == setSize);
  fclose(fin);
  if (insize == 0) {printf("ERROR: input file is empty\n");  exit(-1);}

  // Create prefix array
  long* prefix = new long [insize];

  // Timer - Updated for cross-platform compatibility
  auto start = std::chrono::high_resolution_clock::now();

  // 2. Populate prefix sum array
  for (long cur = 0; cur < insize; cur++) {
    if (input[cur].dis == 0) {
      prefix[cur] = 2;
    }
    else {
      prefix[cur] = (int)input[cur].len + 2 + 1;
    } 
  }

  // 3. Compute prefix sum array
  for (long i = 0; i < insize - 1; i++) {
    prefix[i + 1] += prefix[i];
  }

  // Create parent array
  long* parent = new long [origLength + offset];  
  // Create output array
  byte* const output = new byte [origLength];

  // 4. Use prefix sum to populate parent array
  for (long i = 0; i < insize; i++) {
    const long start = (i == 0) ? 0 : prefix[i - 1];
    if (input[i].dis == 0) { // no matched values
      parent[start + offset] = input[i].val;
      parent[start + offset + 1] = input[i].len;
    } else { // matched values
      for (long j = start; j < prefix[i] - 1; j++) {
        parent[j + offset] = j + offset - input[i].dis;
      }
      parent[prefix[i] + offset - 1] = input[i].val;
    }
  }

  // 5. Populate output by union find
  for (long i = offset; i < origLength + offset; i++) {
    if (parent[i] < offset) {
      output[i - offset] = parent[i];
    }
    else {
      const long parentIndex = find(i, parent);
      output[i - offset] = parent[parentIndex];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("CPU runtime: %.6f s\n", elapsed.count());

  // Write output
  FILE* const fout = fopen(argv[2], "wb");  assert(fout != NULL);
  size = fwrite(output, sizeof(byte), origLength, fout);  assert(size == origLength);
  fclose(fout);

  // Clean up
  delete [] input;
  delete [] output;
  delete [] parent;
  delete [] prefix;
  return 0;
}