#Tests the speed for matrix multiplication in CPU, GPU, and GPU_batched functions.
#Only testing for square matrix, not rectangular matrix.
#Testing number of matrices.


using CuArrays
using Base.Test

include("gemm-test.jl")
# for i in 1:length(c)
#     C[i] = (alpha*A[i]) * B[i] + beta*C[i]
#     println(C[i])
#     h_C = collect(c[i])
#     println(h_C)
#     @test C[i] ≈ h_C atol=1e-10
# end
# nrep::Int64, f::Function,x...; result::Bool=false

function run_cpu_gemm(batchsize::Int64,
                      alpha::Float64,
                      A::Array{Array{Float64,2},1},
                      B::Array{Array{Float64,2},1},
                      beta::Float64,
                      C::Array{Array{Float64,2},1})
    for i = 1:batchsize
        Base.LinAlg.BLAS.gemm!('N','N', alpha, A[i],B[i],beta,C[i])
    end
end

function run_gpu_gemm(batchsize::Int64,
                        alpha::Float64,
                        A::Array{CuArray{Float64,2},1},
                        B::Array{CuArray{Float64,2},1},
                        beta::Float64,
                        C::Array{CuArray{Float64,2},1})
    for i = 1:batchsize
        CuArrays.BLAS.gemm!('N','N', alpha, A[i],B[i],beta,C[i])
    end
end

alpha = rand(Float64)
beta = rand(Float64)
# m = 3
# k = 4
# n = 5
#how many matrices are we calculating:
batch_size = [10,100,500,1000,5000,10000,50000,100000,500000]
#how big are these square matrices
matrix_size = [8,32,128,265,512]
#how many rounds do you want benchmark to run
rounds = 10

file = open("gemm_batched_benchmark_result.csv", "w")
for count in batch_size
    for msize in matrix_size

        A = [rand(Float64,msize,msize) for i in 1:count]
        B = [rand(Float64,msize,msize) for i in 1:count]
        C = [rand(Float64,msize,msize) for i in 1:count]

        a1 = CuArray{Float64, 2}[]
        b1 = CuArray{Float64, 2}[]
        c1 = CuArray{Float64, 2}[]

        a = CuArray{Float64, 2}[]
        b = CuArray{Float64, 2}[]
        c = CuArray{Float64, 2}[]

        for i in 1:length(A)
            push!(a1, CuArray(A[i]))
            push!(b1, CuArray(B[i]))
            push!(c1, CuArray(C[i]))

            push!(a, CuArray(A[i]))
            push!(b, CuArray(B[i]))
            push!(c, CuArray(C[i]))
        end

        CuArrays.CUSOLVER.potrf_batched!('L',a)

        # #using CPU to calculate:
        # cpu_result = benchmark(rounds, run_cpu_gemm, count,alpha, A,B,beta,C)
        #
        # #using GPU to calculate:
        # gpu_result = benchmark(rounds,run_gpu_gemm, count,alpha, a1,b1,beta,c1)
        #
        # #using GPU_batched to calculate:
        # gpu_batched_result = benchmark(rounds, CuArrays.BLAS.gemm_batched!,'N','N', alpha,a,b,beta,c)
        #
        # SpeedUpToCPU = cpu_result[3]/gpu_batched_result[3]
        # SpeedUpToGPU = gpu_result[3]/gpu_batched_result[3]
        #
        # file = open("gemm_batched_benchmark_result.csv", "a")
        # write(file, "Batch_size,$count, Matrix_size,$msize,CPU_result,$(cpu_result[3]),GPU_result,$(gpu_result[3]),GPU_batched_result,$(gpu_batched_result[3]),SpeedUpToCPU,$SpeedUpToCPU,SpeedUpToGPU,$SpeedUpToGPU \n");
        # close(file)
        # #CuArrays.BLAS.gemm_batched!('N', 'N', alpha, a, b, beta, c)
        #
        # #checking correctness
        # for i in 1:length(c)
        #     # C[i] = (alpha*A[i]) * B[i] + beta*C[i]
        #     # println("CPU: C[i] ",C[i])
        #     h_c1 = collect(c1[i])
        #     # println("GPU: c1 ",h_c1)
        #     h_C = collect(c[i])
        #     # println("GPU_BATCHED: h_C ",h_C)
        #     @test C[i] ≈ h_c1 atol=1e-10
        #     @test C[i] ≈ h_C atol=1e-10
        #
        # end
    end
end
println("done")
close(file)
