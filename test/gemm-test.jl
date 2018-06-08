using CuArrays
using Base

include("../src/benchmark.jl")
function matrix_mult(a::Array{Float64, 2}, b::Array{Float64, 2})
    res = a'*b
    return res
end
function matrix_mult(a::CuArray{Float64, 2}, b::CuArray{Float64, 2})
    res = a'*b
    return res
end

#=
file = open("gemm_benchmark_result.csv", "w")
for m in [1024, 2048, 4096, 8192, 16384]
    for n in [128, 256, 512, 1024, 2048]
        if(m>n)
            file = open("gemm_benchmark_result.csv", "a")
            println("m = $m, n = $n")

            srand(123);

            A = randn(m,n)
            B = randn(m,n)
            a = CuArray(A)
            b = CuArray(B)
            #println("A : ", A)
            #println("B : ", B)

            tic(); C = A'*B; toc()
            tic(); c = a'*b; toc()

            #convert GPU array back to host and check result
            h_c = convert(Array{Float64,2},c)
            println("Compare result: ", isapprox(C,h_c; atol = 1e-10))

            #run benchmark
            cpu_result = benchmark(100, Base.LinAlg.BLAS.gemm,'T','N', 1.0, A,B)
            gpu_result = benchmark(100, CuArrays.BLAS.gemm,'T','N', 1.0, a,b)
            speedup = cpu_result[3]/gpu_result[3]
            write(file, "$m, $n, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
            close(file)
        end
    end
end
close(file)
=#
