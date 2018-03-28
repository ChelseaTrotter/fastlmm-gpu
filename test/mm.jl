using CuArrays
#using BenchmarkTools



function matrix_mult(a::Array{Float64, 2}, b::Array{Float64, 2})
    res = a'*b
    return res
end
function matrix_mult(a::CuArray{Float64, 2}, b::CuArray{Float64, 2})
    res = a'*b
    return res
end

#using tic toc in a for loop
function benchmark(nrep::Int64, f::Function,x...; result::Bool=false)
    res = Array{Float64}(nrep)
    for i=1:nrep
        tic()
        f(x...)
        res[i] = toq()
    end
    if(result)
        return res
    else
        return  [minimum(res) quantile(res,[0.25  0.5 0.75]) maximum(res)]
    end
end


srand(123);
m = 10240;
n = 1280;
A = randn(m,n)
B = randn(m,n)
a = CuArray(A)
b = CuArray(B)
#println("A : ", A)
#println("B : ", B)

tic(); C = A'*B; toc()
tic(); c = a'*b; toc()

#println("C: ",C)
#println("C: ",C)

#h_a = convert(Array{Float64,2},a)
h_c = convert(Array{Float64,2},c)
#println("Compare B: ", isapprox(A,h_a; atol = 1e-10))
println("Compare C: ", isapprox(C,h_c; atol = 1e-10))

file = open("benchmark_result.csv", "w")
for m in [4096]
    for n in [8192]
        file = open("benchmark_result.csv", "a")
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
        cpu_result = benchmark(20, matrix_mult,A,B)
        gpu_result = benchmark(20, matrix_mult,a,b)
        speedup = cpu_result[3]/gpu_result[3]
        write(file, "$m, $n, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
        close(file)
    end
end
close(file)
