function benchmark(nrep::Int64, f::Function,x...; result::Bool=false)

    for i in [1024,2048,4096,8192,16384]
        for j in [128, 265, 512, 1024, 2048]
            if(i>j)
                println("n = $i, p = $j")
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
        end
    end
end
