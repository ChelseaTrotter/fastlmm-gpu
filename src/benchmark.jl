function benchmark(nrep::Int64, f::Function,x...; result::Bool=false)

    res = Array{Float64}(nrep)
    for i=1:nrep
        tic()
        f(x...)
        res[i] = toq()
    end
    if(result)
        println(res)
    else
        println([minimum(res) quantile(res,[0.25  0.5 0.75]) maximum(res)])
    end
end
