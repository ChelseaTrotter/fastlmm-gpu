using CuArrays
using Base.Test

alpha = rand(Float64)
beta = rand(Float64)

m = 3
k = 4
n = 5


A = [rand(Float64,m,k) for i in 1:10]
B = [rand(Float64,k,n) for i in 1:10]
C = [rand(Float64,m,n) for i in 1:10]

a = CuArray{Float64, 2}[]
b = CuArray{Float64, 2}[]
c = CuArray{Float64, 2}[]

for i in 1:length(A)
    push!(a, CuArray(A[i]))
    push!(b, CuArray(B[i]))
    push!(c, CuArray(C[i]))
end

CuArrays.BLAS.gemm_batched!('N', 'N', alpha, a, b, beta, c)

for i in 1:length(c)
    C[i] = (alpha*A[i]) * B[i] + beta*C[i]
    println(C[i])
    h_C = collect(c[i])
    println(h_C)
    @test C[i] ≈ h_C atol=1e-10
end
