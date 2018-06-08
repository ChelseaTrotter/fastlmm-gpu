#Tests the speed for matrix multiplication in CPU, GPU, and GPU_batched functions.
#Only testing for square matrix, not rectangular matrix.
#Testing number of matrices.


using CuArrays
using Base.Test

import Base.inv
using Base.LinAlg

include("../src/benchmark.jl")

function run_gpu_cholesky(uplo::Char,
                            A::Array{Array{Float64,2},1},
                            B::Array{Array{Float64,2},1})
    a1 = CuArray{Float64, 2}[]
    b1 = CuArray{Float64, 2}[]
    # c1 = CuArray{Float64, 2}[]

    for i in 1:length(A)
        push!(a1, CuArray(A[i]))
        push!(b1, CuArray(B[i]))
        # push!(c1, CuArray(C[i]))
    end

    for i in 1:length(A)
      CuArrays.CUSOLVER.potrf!(uplo,a1[i])
      CuArrays.CUSOLVER.potrs!(uplo,a1[i],b1[i])
    end
    b1
end

function run_cpu_cholesky(uplo::Char,
                          A::Array{Array{Float64,2},1},
                          B::Array{Array{Float64,2},1})
    a_copy = deepcopy(A)
    b_copy = deepcopy(B)

    for i in 1:length(a_copy)
        Base.LinAlg.LAPACK.potrf!(uplo, a_copy[i])
        Base.LinAlg.LAPACK.potrs!(uplo, a_copy[i], b_copy[i])
    end
    b_copy
end

function run_gpu_batched(uplo::Char,
                        A::Array{Array{Float64,2},1},
                        B::Array{Array{Float64,2},1})

    a = CuArray{Float64, 2}[]
    b = CuArray{Float64, 2}[]
    # c = CuArray{Float64, 2}[]

    for i in 1:length(A)
        push!(a, CuArray(A[i]))
        push!(b, CuArray(B[i]))
    end

    CuArrays.CUSOLVER.potrf_batched!(uplo, a)
    CuArrays.CUSOLVER.potrs_batched!(uplo, a, b)
    b
end

function runtest()
    #how many matrices are we calculating:
    batch_size = [10,100,500,1000,5000,10000,50000,100000,500000]
    #how big are these square matrices
    matrix_size = [32,64,128,256]
    #how many rounds do you want benchmark to run
    rounds = 20

    file = open("cholesky_batched_benchmark_result.csv", "w")
    for count in batch_size
        for msize in matrix_size

            A = [rand(Float64,msize,msize) for i in 1:count]
            B = [rand(Float64,msize,msize) for i in 1:count]
            # C = [rand(Float64,msize,1) for i in 1:count]

            for i in 1:length(A)
                A[i] = A[i]*A[i]'
            end



            #benchmark function will return a tuple,
            #first element is the return value of the benchmarked function,
            #second value is the speed result of benchmarking.

            # using GPU to calculate
            gpu_result_speed = benchmarkWIthReturnValue(rounds, run_gpu_cholesky, 'L', A, B)

            # #using GPU batched function
            gpu_batched_result_speed = benchmarkWIthReturnValue(rounds, run_gpu_batched, 'L', A, B)

            #using CPU to calculate
            cpu_result_speed = benchmarkWIthReturnValue(rounds, run_cpu_cholesky, 'L', A, B)

            # println("*************** B results ******************")
            for i in 1:length(B)
                # println("iter: $(i)")
                h_b1 = collect(gpu_result_speed[1][i])
                h_b = collect(gpu_batched_result_speed[1][i])
                cpu_b = cpu_result_speed[1][i]

                # println("comparing B...")
                # println("GPU result:", h_b1)
                # println("GPU batch resutl:", h_b)
                # println("CPU result:", cpu_b)
                #
                # println("Compare result CPU VS GPU: ", h_b1 ≈ cpu_b)
                # println("Compare result CPU VS BATCHED: ", h_b ≈ cpu_b)

            end
            SpeedUpToCPU = cpu_result_speed[2][3]/gpu_batched_result_speed[2][3]
            SpeedUpToGPU = gpu_result_speed[2][3]/gpu_batched_result_speed[2][3]

            file = open("cholesky_batched_benchmark_result.csv", "a")
            write(file, "Batch_size,$count, Matrix_size,$msize,CPU_result,$(cpu_result_speed[2][3]),
                        GPU_result,$(gpu_result_speed[2][3]),GPU_batched_result,$(gpu_batched_result_speed[2][3]),
                        SpeedUpToCPU,$SpeedUpToCPU,SpeedUpToGPU,$SpeedUpToGPU \n");
            close(file)
        end
    end
    println("done")
    close(file)
end
