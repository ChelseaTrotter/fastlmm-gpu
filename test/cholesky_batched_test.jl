#Tests the speed for matrix multiplication in CPU, GPU, and GPU_batched functions.
#Only testing for square matrix, not rectangular matrix.
#Testing number of matrices.


using CuArrays
using Base.Test

import Base.inv
using Base.LinAlg

include("../src/benchmark.jl")

function run_gpu_potrf(uplo::Char,
                          a::Array{Array{Float64,2},1})
    for i in 1:length(a)
      CuArrays.CUSOLVER.potrf!(uplo,a[i])
    end
end

function run_cpu_potrf(uplo::Char,
                          A::Array{Array{Float64,2},1})
    for i in 1:length(A)
      Base.LinAlg.LAPACK.potrf!(uplo, A[i])
    end
end


function run_gpu_potrs(uplo::Char,
                          a::Array{Array{Float64,2},1},
                          b::Array{Array{Float64,2},1})
    for i in 1:length(a)
      CuArrays.CUSOLVER.potrs!('L',a[i],b[i])
    end
end

function run_cpu_potrs(uplo::Char,
                          A::Array{Array{Float64,2},1},
                          B::Array{Array{Float64,2},1})
    for i in 1:length(A)
      Base.LinAlg.LAPACK.potrs!('L', A[i], B[i])
    end
end


#how many matrices are we calculating:
batch_size = [10#=,100,500,1000,5000,10000,50000,100000,500000=#]
#how big are these square matrices
matrix_size = [8#=,32,128,265,512=#]
#how many rounds do you want benchmark to run
rounds = 10

file = open("cholesky_batched_benchmark_result.csv", "w")
for count in batch_size
    for msize in matrix_size

        A = [rand(Float64,msize,msize) for i in 1:count]
        B = [rand(Float64,msize,1) for i in 1:count]
        C = [rand(Float64,msize,1) for i in 1:count]

        for i in 1:length(A)
            A[i] = A[i]*A[i]'
        end

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



        #using GPU to calculate
        gpu_potrf_speed = benchmark(rounds, run_gpu_potrf, 'L', a1)
        gpu_potrs_speed = benchmark(rounds, run_gpu_potrs, 'L', a1, b1)

        #using GPU batched function
        gpu_batched_potrf_speed = benchmark(rounds, CuArrays.CUSOLVER.potrf_batched!, 'L', a)
        gpu_batched_potrs_speed = benchmark(rounds, CuArrays.CUSOLVER.potrs_batched!, 'L', a, b)

        #using CPU to calculate
        cpu_potrf_speed = benchmark(rounds, run_cpu_potrf, 'L', A)
        cpu_potrs_speed = benchmark(rounds, run_cpu_potrs, 'L', A, B)


        println("*************** B results ******************")
        for i in 1:length(B)
            println("iter: $(i)")
            h_a1 = collect(a1[i])
            h_a = collect(a[i])

            h_b1 = collect(b1[i])
            h_b = collect(b[i])
            println("comparing B...")
            # println("GPU result:", h_b1)
            # println("GPU batch resutl:", h_b)

            println("Compare result CPU VS GPU: ", h_b1 ≈ B[i])
            println("Compare result CPU VS BATCHED: ", h_b ≈ B[i])

        end
    end
end
println("done")
close(file)
