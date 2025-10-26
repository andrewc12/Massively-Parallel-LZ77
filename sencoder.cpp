/*
LZ77 coder simplified for easier parallelization
Serial version 1.2.1 - Largest Match Condition

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
#include <sys/time.h>

typedef unsigned char byte;

static const int MAXLEN = 256;

struct triple {
  byte dis; //distance to match
  byte len; //match length
  byte val; //next value
};

int main(int argc, char* argv[])
{
  printf("LZ77 encoder (%s)\n", __FILE__);
  if (argc != 3) {printf("USAGE: %s input_file_name output_file_name\n", argv[0]);  exit(-1);}

  // Read input
  FILE* const fin = fopen(argv[1], "rb");  assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  long size = ftell(fin);  assert(size > 0);
  byte* const input = new byte [size];
  fseek(fin, 0, SEEK_SET);
  const long insize = fread(input, sizeof(byte), size, fin);  assert(insize == size);
  fclose(fin);
  if (insize == 0) {printf("ERROR: input file is empty\n");  exit(-1);}

  // Create output
  triple* const output = new triple [insize];  // upper bound
  long outsize = 0;
  
  // Timer
  timeval start, end;
  gettimeofday(&start, NULL);
  
  //1. Iterate through input
  long cur = 0;
  while (cur < insize) {
    //2. Find Matches
    long maxlen = 0;
    long maxidx;
    long pos = cur - 1;
    while ((cur - pos < MAXLEN) && (pos >= 0)) {
      if (input[pos] == input[cur]) {
        long len = 1;
        //3. Find longest match
        while ((len <= MAXLEN) && (cur + len < insize) && (input[pos + len] == input[cur + len])) {
          len++;
        }
        if (maxlen <= len) { //will save the largest or farthest within the window
          maxlen = len;
          maxidx = pos;
        }
      }
      pos--;
    }

    //4. Store triple in output
    if (maxlen < 2) {
      output[outsize].dis = 0;
      output[outsize].val = (cur < insize) ? input[cur] : 0;
      cur++;
      output[outsize].len = (cur < insize) ? input[cur] : 0;
      cur++;
    } else {
      output[outsize].dis = cur - maxidx;
      output[outsize].len = maxlen - 2;
      cur += maxlen;
      output[outsize].val = (cur < insize) ? input[cur] : 0;
      cur++;
    }
    outsize++;
  }
  gettimeofday(&end, NULL);
  printf("CPU runtime: %.6f s\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);

  // Write output
  FILE* const fout = fopen(argv[2], "wb");  assert(fout != NULL);
  size = fwrite(&insize, sizeof(long), 1, fout);  assert(1 == size);
  size = fwrite(output, sizeof(triple), outsize, fout);  assert(outsize == size);
  fclose(fout);

  // Print compression ratio
  printf("compression ratio: %.3f\n", 1.0 * insize / (sizeof(long) + sizeof(triple) * outsize));

  // Clean up
  delete [] input;
  delete [] output;
  return 0;
}