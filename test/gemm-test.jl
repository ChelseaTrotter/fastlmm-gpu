using CuArrays
using Base
using BenchmarkTools

include("../src/benchmark.jl")

alf = 1.0
bet = 0.0
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

file = open("gemm_benchmark_result.csv", "w")
# for m in [1024, 2048, 4096, 8192, 16384]
    # for n in [128, 256, 512, 1024, 2048]
for m in [5120]
    for n in [5120]
        if(m>=n)
            file = open("gemm_benchmark_result.csv", "a")
            println("m = $m, n = $n")

            srand(123);

            A = randn(m,n);
            B = randn(m,n);
            C = similar(A);
            # a = CuArray(A);
            # b = CuArray(B);
            #println("A : ", A)
            #println("B : ", B)

            println("CPU runtime:")
            tic(); cpu_run(A,B,C);toc();
            println("GPU runtime:")
            tic(); gpu_run(A,B,C);toc();

            #convert GPU array back to host and check result
            # h_c = convert(Array{Float64,2},c)
            # println("Compare result: ", isapprox(C,h_c; atol = 1e-5))

            #run benchmark
            # cpu_result = @btime LinAlg.BLAS.gemm('T','N', $A,$B)
            # gpu_result = @btime CuArrays.BLAS.gemm('T','N', $a,$b)
            # speedup = cpu_result/gpu_result
            # write(file, "$m, $n, $(cpu_result),  $(gpu_result), $speedup\n");
            # close(file)

            cpu_result = benchmark(10, cpu_run, A,B,C)
            gpu_result = benchmark(10, gpu_run,A,B,C)
            speedup = cpu_result[3]/gpu_result[3]
            println(cpu_result)
            println(gpu_result)
            write(file, "testing gemm in julia. 
                    Matrix size: 5120 double precision. 
                    Comparing to C result: GPU 0.511 seconds , CPU (using cblas_dgemm)16.19 seconds.
                    Does not include data transfer time\n")
            write(file, "$m, $n, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
            close(file)
        end
    end
end
close(file)
