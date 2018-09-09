setlocal
set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\nvcc.exe" lenet.cu --cudart static --compile --machine 64 -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" -I "."
