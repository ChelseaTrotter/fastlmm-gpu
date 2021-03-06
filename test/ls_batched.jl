#######
# least squares
#######

using CuArrays
using LinearAlgebra

include("../src/benchmark.jl")

mutable struct Wls
    b::Array{Float64,2}
    sigma2::Float64
    ell::Float64
end

"""
ls Batched version
y = outcome, matrix
X = predictors, matrix

The variance estimate is maximum likelihood
"""
function cholesky_gpu(uplo::Char, A::Array{CuArray{Float64, 2}, 1}, B::Array{CuArray{Float64, 2}, 1})
    # d_a = CuArray{Float64, 2}[]
    # d_b = CuArray{Float64, 2}[]

    # for i in 1:length(A)
    #     push!(d_a, CuArray(A[i]) )
    #     push!(d_b, CuArray(B[i]) )
    # end

    for i in 1:length(A)
        CuArrays.CUSOLVER.potrf!(uplo,A[i])
        CuArrays.CUSOLVER.potrs!(uplo,A[i],B[i])
    end

    return B
end

function cholesky_batched(uplo::Char, A::Array{CuArray{Float64, 2}, 1}, B::Array{CuArray{Float64, 2}, 1})

    CuArrays.CUSOLVER.potrf_batched!(uplo, A)
    CuArrays.CUSOLVER.potrs_batched!(uplo, A, B)
    return B
end

function cholesky_cpu(uplo::Char, A::Array{Array{Float64, 2}, 1}, B::Array{Array{Float64, 2}, 1})
    a = deepcopy(A)
    b = deepcopy(B)

    for i in 1:length(a)
        LinearAlgebra.LAPACK.potrf!(uplo, a[i])
        LinearAlgebra.LAPACK.potrs!(uplo, a[i], b[i])
    end
    return b
end


function solveleq(uplo::Char, A::Array{Array{Float64,2}, 1}, B::Array{Array{Float64,2}, 1})

        return cholesky_cpu(uplo, A,B)

end

function solveleq(uplo::Char, A::Array{CuArray{Float64,2}, 1}, B::Array{CuArray{Float64,2}, 1},  hwChoice = "GPU_BATCHED")

    # println("You are using GPU batched function for least square, only cholesky decomposition is currently supported. QR or LU are not supported for batched process.")
    # println("Doing cholesky algorithm...")

    if(hwChoice == "GPU")
        return cholesky_gpu(uplo,A,B)

    else(hwChoice == "GPU_BATCHED")
        return cholesky_batched(uplo,A,B)

    end
end


#CPU version
function ls(y::Array{Array{Float64,2},1}, X::Array{Array{Float64,2}, 1}, loglik=false)
    # number of individuals
    n = size(y[1],1)
    # number of covariates
    p = size(X[1],2)

    batchsize = length(y)

    wls_array = Array{Wls}(undef,batchsize)

    XtX_array = Array{Array{Float64,2},1}(undef,batchsize)
    Xty_array = Array{Array{Float64,2},1}(undef,batchsize)

    for i in 1:length(y)
        XtX_array[i] = transpose(X[i]) * X[i]
        Xty_array[i] = transpose(X[i]) * y[i] 
    end

    b_array = solveleq('L', XtX_array, Xty_array)

    for i in 1:batchsize
        yhat = X[i]*b_array[i]
        rss = norm((y[i]-yhat))^2
        sigma2 = rss/n

        # return coefficient and variance estimate
        logdetSigma = n*log(sigma2)
        ell = -0.5 * ( logdetSigma + n )
        wls_array[i] = Wls(b_array[i], sigma2, ell)

    end

    return wls_array

end


#GPU version and batched version.
function ls(y::Array{CuArray{Float64,2}, 1},X::Array{CuArray{Float64,2}, 1}, batched="GPU_BATCHED", loglik=false)

    # number of individuals
    n = size(y[1],1)
    # number of covariates
    p = size(X[1],2)

    batchsize = length(y)
    wls_array = Array{Wls}(undef, batchsize)
    XtX_array = Array{CuArray{Float64,2},1}(undef, batchsize)
    Xty_array = Array{CuArray{Float64,2},1}(undef, batchsize)
    for i in 1:length(y)
        XtX_array[i] = transpose(X[i]) * X[i]
        Xty_array[i] = transpose(X[i]) * y[i]
    end

    if batched == "GPU_BATCHED"
        b_array = solveleq('L', XtX_array, Xty_array, "GPU_BATCHED")
    else
        b_array = solveleq('L', XtX_array, Xty_array, "GPU")
    end
    for i in 1:length(y)
        yhat = X[i]*b_array[i] #use gpu
        rss = norm((y[i]-yhat))^2
        sigma2 = rss/n

        # return coefficient and variance estimate
        logdetSigma = n*log(sigma2)
        ell = -0.5 * ( logdetSigma + n )
        wls_array[i] = Wls(b_array[i], sigma2, ell)

    end

    return wls_array

end

# function runtest()
    file = open("ls_batched_result.csv", "w")

    batchsizes = [100,#=10000,50000,100000,500000=#]
    #matrix_size = [8#=,32,64,128=#]

    for bs in batchsizes
        for n in [32#=,64,128,256,512,1024,2048=#]
            for p in [2#=,4,8,16,32,64=#]
                if (n >= p)
                    X_array = Array{Array{Float64,2},1}(undef, bs)
                    Y_array = Array{Array{Float64,2},1}(undef, bs)

                    d_x = Array{CuArray{Float64, 2},1}(undef, bs)
                    d_y = Array{CuArray{Float64, 2},1}(undef, bs)

                    println("n = $n, p = $p")

                    b = ones(p,1)
                    for i in 1:bs
                        X = randn(n*2, p);
                        Y = X*b + randn(n*2, 1)
                        X_array[i] = X
                        Y_array[i] = Y
                        d_x[i] = CuArray(X)
                        d_y[i] = CuArray(Y)
                    end

                    time_ns(); cpu_result = ls(Y_array,X_array); time_ns()
                    @elapsed gpu_batched_result = ls(d_y, d_x);
                    @elapsed gpu_result = ls(d_y,d_x, "GPU"); 


                    # printing out correctness
                    for i in 1:bs
                         println("Compare result with batched: ", isapprox(cpu_result[i].b,convert(Array{Float64,2}, gpu_batched_result[i].b)))
                         println("Compare result with gpu: ", isapprox(cpu_result[i].b,convert(Array{Float64,2}, gpu_result[i].b)))
                    end

                    # cpu_speed = benchmark(20, ls, Y_array,X_array)
                    # gpu_speed = benchmark(20, ls, d_y, d_x, "GPU")
                    # gpu_batched_speed = benchmark(20, ls, d_y, d_x)
                    #
                    # # println(cpu_speed)
                    # # println(gpu_speed)
                    #
                    # cpu_vs_gpu = cpu_speed[3]/gpu_speed[3]
                    # cpu_vs_batched = cpu_speed[3]/gpu_batched_speed[3]
                    # gpu_vs_batched = gpu_speed[3]/gpu_batched_speed[3]
                    #
                    #
                    # println("bs:$bs,n:$n, p: $p, CPU: $(cpu_speed[3]), GPU: $(gpu_speed[3]),
                    # cpu vs gpu: $cpu_vs_gpu, cpu vs batched: $cpu_vs_batched, gpu vs batched: $gpu_vs_batched\n");
                    #
                    # write(file, "$bs, $n, $p, $(cpu_speed[3]), $(gpu_speed[3]), Speedup, $cpu_vs_gpu, $cpu_vs_batched, $gpu_vs_batched\n");
                end
            end
        end
    end
    close(file)
# end
