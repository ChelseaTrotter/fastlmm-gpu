using CuArrays
using Base
using BenchmarkTools
using LinearAlgebra


include("../src/benchmark.jl")

const alf = 1.0
const bet = 0.0
const gb =  1073741824

# gemm!(tA, tB, alpha, A, B, beta, C)

function cpu_run(a::Array, b::Array, c::Array)
    return LinAlg.BLAS.gemm!('N','N',alf, a,b, bet, c)
end

# CuArrays.CUBLAS.gemm!('N','N',alpha,d_A,d_B,beta,d_C1)

function gpu_run(a::Array, b::Array, c::Array)
    A = CuArray(a)
    B = CuArray(b)
    C = CuArray(c)
    CuArrays.BLAS.gemm!('N', 'N', alf, A,B, bet, C)
    return collect(C)
end

function get_max_doubles()
    gpu_mem_size = 2
    size_of_double_float = 8 #a double floating point number takes 8 bytes to store

    if gethostname() == "cuda-linux"
        gpu_mem_size = 2 - 0.3
    else 
        gpu_mem_size = 16 - 0.5
    end
    println("Total GPU memory size: $gpu_mem_size GB. \n")
    return (gpu_mem_size*gb)/size_of_double_float
end


dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
host = gethostname()

file = open("./gemm-timing/dgemm-result@$host@$dt_now.csv", "w")

thread_counts = [1,16,40,80]
for t in thread_counts
    LinearAlgebra.BLAS.set_num_threads(t)
    core_nums = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    write(file, "number of cores/threads used in openblas: $core_nums");

    # msizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; # range from 1k to 1m in log scale
    # nsizes = msizes;
    # psizes = [16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]; # ranges from 10 to 4k in log scale. 

    # for m in msizes
    #     for p in psizes
    #         for n in nsizes
    #             file = open("./gemm-timing/dgemm-result@$host@$dt_now.csv", "a")
    #             total_doubles = (m*n + n*p + m*p)
    #             if ((m*n + n*p + m*p)>get_max_doubles())
    #             # if (1>0)
    #                 println("Matrices are too big to fit in GPU memory. Skipping this configuration. M is $m, N is $n, P is $p");
    #                 write(file, "Matrices are too big to fit in GPU memory. Skipping this configuration. M is $m, N is $n, P is $p\n");
    #                 close(file);
    #             else
    #                 println("m = $m, n = $n, p: $p\n")
    #                 srand(123);

    #                 #generating double precision matrix
    #                 A = randn(m,n);
    #                 B = randn(n,p);
    #                 C = zeros(m,p);

    #                 println("CPU runtime:")
    #                 tic(); cpu_run(A,B,C);toc();
    #                 println("GPU runtime:")
    #                 tic(); gpu_run(A,B,C);toc();

    #                 cpu_result = benchmark(10, cpu_run, A,B,C)
    #                 gpu_result = benchmark(10, gpu_run,A,B,C)
    #                 speedup = cpu_result[3]/gpu_result[3]
    #                 println(cpu_result)
    #                 println(gpu_result)
    #                 # write(file, "testing double precision gemm in julia. Does include data transfer time
    #                 # m, n, (cpu_result[3]),  (gpu_result[3]), speedup\n")
    #                 mem_req_gb = (total_doubles * 8 ) / gb
    #                 write(file, "$m, $n, $p, $mem_req_gb, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    #                 close(file)
    #             end
    #         end 
    #     end
    # end
end
file = open("./gemm-timing/dgemm-result@$host@$dt_now.csv", "a")
dt_finish = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
write(file, "Finshied at $dt_finish");
close(file)
