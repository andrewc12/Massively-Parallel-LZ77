# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -arch=sm_60
CXX_FLAGS = 
LIBS = -lcudart

# Targets
TARGETS = pencoder pdecoder sencoder sdecoder

# Default target
all: $(TARGETS)

# Parallel encoder (CUDA)
pencoder: pencoder.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# Parallel decoder (CUDA)
pdecoder: pdecoder.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# Serial encoder (C++)
sencoder: sencoder.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $<

# Serial decoder (C++)
sdecoder: sdecoder.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(TARGETS)

# Phony targets
.PHONY: all clean