using CuArrays
using Base
using BenchmarkTools

include("../src/benchmark.jl")

alf = 1.0f0
bet = 0.0f0

function cpu_run(a::Array, b::Array, c::Array)
    return LinAlg.BLAS.gemm!('N','N',alf, a,b, bet, c)
end

function gpu_run(a::Array, b::Array, c::Array)
    A = CuArray(a)
    B = CuArray(b)
    C = CuArray(c)
    CuArrays.BLAS.gemm!('N', 'N', alf, A,B, bet, C)
    return collect(C)
end

function get_num_singles()
    gb =  1073741824
    gpu_mem_size = 2
    size_of_single_float = 4 #a single floating point number takes 4 butes to store

    if gethostname() == "cuda-linux"
        gpu_mem_size = 2 
    else 
        gpu_mem_size = 16 
    end
    println("Total GPU memory size: $gpu_mem_size GB. \n")
    return (gpu_mem_size*gb)/size_of_single_float

end

dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
host = gethostname()
file = open("./gemm-timing/sgemm-result@$host@$dt_now.csv", "w")

msizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; # range from 1k to 1m in log scale
nsizes = msizes;
psizes = [16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]; # ranges from 10 to 4k in log scale. 

for m in msizes
    for n in nsizes
        for p in psizes 
            file = open("./gemm-timing/sgemm-result@$host@$dt_now.csv", "a")
            if ((m*p + n*p + m*n)>get_num_singles())
                println("Matrices are too big to fit in GPU memory. Skipping this configuration. M is $m, N is $n, P is $p");
                write(file, "Matrices are too big to fit in GPU memory. Skipping this configuration. M is $m, N is $n, P is $p");
                break;
            end
            println("m = $m, n = $n, p: $p\n")
            srand(123);

            A = randn(Float32, (m,n));
            B = randn(Float32, (n,p));
            C = zeros(Float32, (m,p));

            println("CPU runtime:")
            tic(); cpu_run(A,B,C);toc();
            println("GPU runtime:")
            tic(); gpu_run(A,B,C);toc();

            cpu_result = benchmark(10, cpu_run, A,B,C)
            gpu_result = benchmark(10, gpu_run,A,B,C)
            speedup = cpu_result[3]/gpu_result[3]
            println(cpu_result)
            println(gpu_result)
            # write(file, "testing single precision gemm in julia. Does include data transfer time
            # m, n, (cpu_result[3]),  (gpu_result[3]), speedup\n")
            write(file, "$m, $n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
            close(file)
        end
    end
end
close(file)
