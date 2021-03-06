
##################################################################
# ls: least squares
##################################################################

import Base.inv
using CuArrays
using LinearAlgebra



include("benchmark.jl")

type Wls
    b::Array{Float64,2}
    sigma2::Float64
    ell::Float64
end


"""
ls: Weighted least squares estimation

y = outcome, matrix
X = predictors, matrix

The variance estimate is maximum likelihood
"""

function ls(y::Array{Float64,2},X::Array{Float64,2}, loglik=false)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)

    XtX = At_mul_B(X,X)
    b = solveleq(XtX,At_mul_B(X,y))
    # estimate yy and calculate rss
    yhat = X*b
    # yyhat = q*At_mul_B(q,yy)
    rss = norm((y-yhat))^2
    sigma2 = rss/n

    # return coefficient and variance estimate
    logdetSigma = n*log(sigma2)
    ell = -0.5 * ( logdetSigma + n )

    return Wls(b,sigma2,ell)

end

function ls(y::CuArray{Float64,2},X::CuArray{Float64,2}, loglik=false)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)

    XtX = At_mul_B(X,X)
    b = solveleq(XtX,At_mul_B(X,y), "QR") #choose the factorization method here.
    # estimate yy and calculate rss
    yhat = X*b
    # yyhat = q*At_mul_B(q,yy)
    rss = norm((y-yhat))^2
    sigma2 = rss/n

    # return coefficient and variance estimate
    logdetSigma = n*log(sigma2)
    ell = -0.5 * ( logdetSigma + n )

    return Wls(b,sigma2,ell)

end

# function to solve linear equations using Cholesky factorization
function solveleq( A::CuArray{Float64,2}, B::CuArray{Float64,2}, method="CH")
    b = copy(B)
    a = copy(A)
    if(method == "CH")
        CuArrays.CUSOLVER.potrf!('L',a)
        CuArrays.CUSOLVER.potrs!('L',a,b)
    elseif(method == "QR")
        (a, tau) = CuArrays.CUSOLVER.geqrf!(a)
        CuArrays.CUSOLVER.ormqr!('L', 'T', a, tau, b)
        alpha = Float64(1)
        CuArrays.BLAS.trsm!('L', 'U', 'N', 'N', alpha, a, b)
    # else
    #     #default method LU
    #     (a, ipiv) = CuArrays.CUSOLVER.getrf!(a)
    #     b = CuArrays.COSOLVER.getrs!('N', a, ipiv, b)
    end
    return b
end

function solveleq( A::Array{Float64,2}, B::Array{Float64,2}, method="CH")
    a = copy(A)
    b = copy(B)
    if(method == "CH")
        LinearAlgebra
        .LAPACK.potrf!('L', a)
        LinearAlgebra
        .LAPACK.potrs!('L', a, b)
    elseif(method == "QR")
        (a, tau) = LinearAlgebra
        .LAPACK.geqrf!(a)
        LinearAlgebra
        .LAPACK.ormqr!('L', 'T', a, tau, b)
        alpha = Float64(1)
        LinearAlgebra
        .BLAS.trsm!('L', 'U', 'N', 'N', alpha, a, b)
    # else
    #     #default method LU
    #     #x = A\B
    #     b = inv(a) * b
    end
    return b
end

function inv(x::CuArray)
    return CuArrays.BLAS.matinv_batched([x])[2][1]
end


# using CuArrays
# CuArrays.CUSOLVER.potrf!('L',a)

function run_ls()
    file = open("benchmark_result.csv", "w")
    for n in [1024,2048,4096,8192, 16384]
        for p in [128, 265, 512, 1024, 2048]
            if(n>p)

                println("n = $n, p = $p")
                b = ones(p,1);
                X = randn(n*2,p);
                Y = X*b+ randn(n*2,1);
                #W = repeat([4.0; 1.0],inner=n);
                x = CuArray(X);
                y = CuArray(Y);
                #w = CuArray(W);

                tic(); cpu = ls(Y,X);toc()
                tic(); gpu = ls(y,x);toc()

                #convert GPU array back to host and check result
                h_b = convert(Array{Float64,2},gpu.b)
                # println("CPU result: ", cpu.b)
                # println("GPU result: ", h_b)
                println("Compare result: ", isapprox(cpu.b,h_b; atol = 1e-10))

                #run benchmark
                cpu_result = benchmark(100, ls,Y,X)
                gpu_result = benchmark(100, ls,y,x)
                println(cpu_result)
                println(gpu_result)
                speedup = cpu_result[3]/gpu_result[3]
                write(file, "$n, $p, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");


            end
        end
    end
    close(file)
end
