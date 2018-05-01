function benchmark(nrep::Int64, f::Function,x...; result::Bool=false)

    res = Array{Float64}(nrep)
    local returnvalue
    for i=1:nrep
        tic()
        returnvalue = f(x...)
        res[i] = toq()
    end
    if(result)
        return (returnvalue,res)
    else
        return (returnvalue, [minimum(res) quantile(res,[0.25  0.5 0.75]) maximum(res)])
    end
end
