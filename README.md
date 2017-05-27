# Awesome Cuda

This is a list of useful libraries and resources for CUDA
development.


## Libraries

* [ANNETGPGPU](https://github.com/ANNetGPGPU/ANNetGPGPU) is a CUDA based artifical neural network library.
* [AmgX](https://developer.nvidia.com/amgx) is a simple path to accelerated core solvers, providing up to 10x acceleration in the computationally intense linear solver portion of simulations, and is very well suited for implicit unstructured methods.
* [ArrayFire](https://developer.nvidia.com/arrayfire) is a comprehensive, open source GPU function library. Includes functions for math, signal and image processing, statistics, and many more. Interfaces for C, C++, Java, R and Fortran.
* [CUBLAS](https://developer.nvidia.com/cublas) is a GPU-accelerated version of the complete standard BLAS library that delivers 6x to 17x faster performance than the latest MKL BLAS.
* [CUB](https://github.com/NVlabs/cub) is a flexible library of cooperative threadblock primitives and other utilities for CUDA kernel programming.
* [CUDA Math](https://developer.nvidia.com/cuda-math-library) is an industry proven, highly accurate collection of standard mathematical functions, providing high performance on NVIDIA GPUs.
* [CUDNN](https://developer.nvidia.com/cudnn) - A GPU-accelerated library of primitives for deep neural networks providing highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
* [CUDPP](https://github.com/cudpp/cudpp) - A library that provides 15 parallel primitives. In difference to Thrust, CUDPP is a more performance oriented library, and it is also much more low-level. Recommended if performance is more important than programmer productivity.
* [CUFFT](https://developer.nvidia.com/cufft) provides a simple interface for computing FFTs up to 10x faster, without having to develop your own custom GPU FFT implementation.
* [CULA](https://developer.nvidia.com/em-photonics-cula-tools) is a GPU-accelerated linear algebra library by EM Photonics, that utilizes CUDA to dramatically improve the computation speed of sophisticated mathematics.
* [CUSOLVER](https://developer.nvidia.com/cusolver) is a collection of dense and sparse direct solvers which deliver significant acceleration for Computer Vision, CFD, Computational Chemistry, and Linear Optimization applications.
* [CUSPARSE](https://developer.nvidia.com/cusparse) is a library provides a collection of basic linear algebra subroutines used for sparse matrices that delivers over 8x performance boost.
* [GPP](https://developer.nvidia.com/geometric-performance-primitives-gpp) is a computational geometry engine that is optimized for GPU acceleration, and can be used in advanced Graphical Information Systems (GIS), Electronic Design Automation (EDA), computer vision, and motion planning solutions.
* [GUNROCK](http://gunrock.github.io/) is a GPU library for graph analytics that couples an expressive, powerful programming model with a high-performance implementation.
* [Hemi](https://github.com/harrism/hemi) - A nice little utility library that allows you to write code that can be run either on the CPU or GPU, and allows you to launch C++ lambda functions as CUDA kernels. Its main goal is to make it easier to write portable CUDA programs.
* [HiPLAR](https://developer.nvidia.com/hiplar) delivers high performance linear algebra (LA) routines for the R platform for statistical computing using the latest software libraries for heterogeneous architectures.
* [IMSL](https://developer.nvidia.com/imsl-fortran-numerical-library) is a comprehensive set of mathematical and statistical functions that offloads work to GPUs.
* [JCuda](http://www.jcuda.org/) is Java bindings for CUDA.
* [MAGMA](https://developer.nvidia.com/magma) is a collection of next gen linear algebra routines. Designed for heterogeneous GPU-based architectures. Supports current LAPACK and BLAS standards.
* [MVGRAPH](https://developer.nvidia.com/nvgraph) is a GPU-accelerated graph analytics library.
* [Moderngpu](https://github.com/moderngpu/moderngpu) is a productivity library for general-purpose computing on GPUs. It is a header-only C++ library written for CUDA. The unique value of the library is in its accelerated primitives for solving irregularly parallel problems.
* [NPP](https://developer.nvidia.com/npp)  is a GPU accelerated library with a very large collection of 1000's of image processing primitives and signal processing primitives.
* [NVBIO](https://developer.nvidia.com/NVBIO) is a GPU-accelerated C++ framework for High-Throughput Sequence Analysis for both short and long read alignment.
* [NVIDIA VIDEO CODEC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) is an accelerated video compression with the NVIDIA Video Codec SDK.
* [PARALUTION](https://developer.nvidia.com/paralution) is a library for sparse iterative methods with special focus on multi-core and accelerator technology such as GPUs.
* [Parallel Primitives Library: chag::pp](https://newq.net/archived/www.cse.chalmers.se/pub/pp/) - This library provides the parallel primitives Reduction, Prefix Sum, Stream Compaction, Split, and Radix Sort. The authors have [demonstrated](https://newq.net/archived/www.cse.chalmers.se/pub/pp/stream_compaction_pres.pdf) that their implementation of Stream Compaction and Prefix Sum are the fastest ones available!
* [PyCUDA](https://mathema.tician.de/software/pycuda/) lets you access Nvidia‘s CUDA parallel computation API from Python.
* [R+GPU](http://brainarray.mbni.med.umich.edu/brainarray/rgpgpu/) enables GPU Computing in the R Statistical Environment.
* [TensorRT](https://developer.nvidia.com/tensorrt) is a high performance neural network inference library for deep learning applications.
* [Thrust](https://github.com/thrust/thrust) - A parallel algorithms library whose main goal is programmer productivity and rapid development. But if your main goal is reaching the best possible performance, you are advised to use a more low-level library, such as CUDPP or chag::pp.
* [Vexcl](https://github.com/ddemidov/vexcl) is a C++ vector expression template library for OpenCL/CUDA.

## [cuDNN supported frameworks](https://developer.nvidia.com/deep-learning-frameworks)

* [Caffe](http://caffe.berkeleyvision.org/) is a deep learning framework made with expression, speed, and modularity in mind.
* [Caffe2](https://developer.nvidia.com/caffe2) is a deep learning framework enabling simple and flexible deep learning.
* [CNTK](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/)  is a unified deep-learning toolkit from Microsoft Research that makes it easy to train and combine popular model types across multiple GPUs and servers.
* [TensorFlow](http://tensorflow.org/) is a software library for numerical computation using data flow graphs, developed by Google’s Machine Intelligence research organization.
* [theano](http://deeplearning.net/software/theano/) is a math expression compiler that efficiently defines, optimizes, and evaluates mathematical expressions involving multi-dimensional arrays.
* [torch](http://torch.ch/) is a scientific computing framework that offers wide support for machine learning algorithms.
* [mxnet](https://github.com/dmlc/mxnet) is a deep learning framework designed for both efficiency and flexibility that allows you to mix the flavors of symbolic programming and imperative programming to maximize efficiency and productivity.
* [Chainer](http://chainer.org/) is a deep learning framework that’s designed on the principle of define-by-run. Unlike frameworks that use the define-and-run approach, Chainer lets you modify networks during runtime, allowing you to use arbitrary control flow statements.
* [Keras](http://chainer.org/) is a minimalist, highly modular neural networks library, written in Python, and capable of running on top of either TensorFlow or Theano. Keras was developed with a focus on enabling fast experimentation.

## Papers

* [Multireduce and Multiscan on Modern GPUs](http://hiperfit.dk/pdf/marco-eilers-thesis.pdf) - In this
  master's thesis, it is examined how you can implement an efficient
  Multireduce and Multiscan on the GPU.

* [Efficient Parallel Scan Algorithms for Many-core GPUs](http://www.idav.ucdavis.edu/publications/print_pub?pub_id=1041) - In this paper, it is shown how the scan and segmented scan algorithms
  can be efficiently implemented using a divide-and-conquer approach.

* [Ana Balevic's homepage](http://tesla.rcub.bg.ac.rs/~taucet/coding.html) - Ana Balevic has done research in implementing compression
  algorithms on the GPU, and in her publications she describes fast
  implementations of RLE, VLE(Huffman coding) and arithmetic coding on
  the GPU.

* [Run-length Encoding on Graphics
  Hardware](https://www.cs.uaf.edu/media/filer_public/2013/08/27/ms_cs_ruth_rutter.pdf) - Shows another approach to implementing RLE on the GPU. In
  difference to Ana Belvic's fine grain parallelization approach, this
  paper describes an approach where the data is split into blocks,
  and then every thread is assigned a block and does RLE on that block.

* [Efficient Stream Compaction on Wide SIMD Many-Core Architectures](http://www.cse.chalmers.se/~uffe/streamcompaction.pdf) - The paper that the chag::pp library is based on.

* [Histogram calculation in CUDA](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf) - This article explains how a histogram can be calculated in CUDA.

* [Modern GPU](https://nvlabs.github.io/moderngpu/index.html) - Modern GPU
is a text that describes algorithms and strategies for writing fast
CUDA code. And it also provides a library where all of the explained
concepts are implemented.

## Articles

* [GPU Pro Tip: Fast Histograms Using Shared Atomics on
Maxwell](https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/) - In this article, it is shown how an even faster histogram
calculation algorithm can be implemented.

* [Faster Parallel Reductions on
Kepler](https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/) - It is shown in this article how the reduction algorithm described by [Mark
Harris](https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf)
can be made faster on Kepler.

* [GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell](https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/) - It is shown how we can use shared memory atomics to implement a faster histogram implementation on Maxwell.


## Presentations

* [Optimizing Parallel Reduction in CUDA](https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf) - In this presentation it is shown how a fast, but relatively simple, reduction
algorithm can be implemented.

* [CUDA C/C++ BASICS](https://www.olcf.ornl.gov/wp-content/uploads/2013/02/Intro_to_CUDA_C-TS.pdf) - This presentations explains the concepts of CUDA kernels,
memory management, threads, thread blocks, shared memory, thread
syncrhonization. A simple addition kernel is shown, and an optimized stencil
1D stencil kernel is shown.

* [Advanced CUDA - Optimizing to Get 20x
  Performance](https://www.nvidia.com/content/cudazone/download/Advanced_CUDA_Training_NVISION08.pdf) - This presentation covers: Tesla 10-Series Architecture, Particle
  Simulation Example, Host to Device Memory Transfer, Asynchronous
  Data Transfers, OpenGL Interoperability, Shared Memory, Coalesced
  Memory Access, Bank Conflicts, SIMT, Page-locked Memory, Registers,
  Arithmetic Intensity, Finite Differences Example, Texture Memory.

* [Advanced CUDA Webinar - Memory
  Optimizations](http://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf) - This presentation covers: Asynchronous Data Transfers , Context
  Based Synchronization, Stream Based Synchronization, Events, Zero
  Copy, Memory Bandwidth, Coalescing, Shared Memory, Bank Conflicts,
  Matrix Transpose Example, Textures.

* [Better Performance at Lower
  Occupancy](http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) - Excellent presentation where it is shown that we can achieve better
  performance by assigning more parallel work to each thread and by using
  Instruction-level parallelism. Covered topics are:
  Arithmetic Latency, Arithmetic Throughput, Little's Law,
  Thread-level parallelism(TLP), Instruction-level parallelism(ILP),
  Matrix Multiplication Example.

* [Fun With Parallel Algorithms. Segmented Scan. Neutral territory method](http://www.cs.cmu.edu/afs/cs/academic/class/15418-s12/www/lectures/24_algorithms.pdf) - In these slides, it is shown how a segmented scan can easily be implemented using a variation of a normal scan.

* [GPU/CPU Programming for Engineers - Lecture 13](http://www.ce.jhu.edu/dalrymple/classes/602/Class13.pdf) - This lecture provides a good walkthrough of all the different memory types: Global Memory, Texture Memory, Constant Memory, Shared Memory, Registers and Local Memory.

## Videos

* [Intro to Parallel Programming CUDA -
  Udacity](https://www.youtube.com/playlist?list=PLGvfHSgImk4aweyWlhBXNF6XISY3um82_) - An Udacity course for learning CUDA.

## Contributing

This list is still under construction and is far from done. Anyone who
wants to add links to the list are very much welcome to do so by a
pull request!

