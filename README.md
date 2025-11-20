Parallel and Serial encoders and decoders for LZ77 by Kayla Wesley as part of her masters thesis Massively Parallel LZ77 Compression and Decompression on the GPU available at <https://hdl.handle.net/10877/16093>


# Build

## Linux
```bash
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

## Windows
```bat
%comspec% /k "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
rmdir /s /q build
mkdir build
cd build
cmake ..
cmake --build . --config Release --parallel
#msbuild LZ77Compression.sln
cd ..
```
