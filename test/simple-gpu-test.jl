using CuArrays, CUDAnative

using Test

function vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end



dims = (3,4)
a = [1 2 3 4 ; 5 6 7 8 ; 9 10 11 12]
d_a = CuArray(a)

println("size of CuArray d_a : $(sizeof(d_a))")
ndrange = prod(dims)
threads=12
blocks = max(Int(ceil(ndrange/threads)), 1)
println("blocks is $blocks")
@cuda blocks=blocks threads=threads square(d_a)
result = Array(d_a)
display(result)
@test a.*a ≈ result

function square(a) 
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i <  ndrange)
        a[i] = a[i] * a[i]
    end
    return 
end

