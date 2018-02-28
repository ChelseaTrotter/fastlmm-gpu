######################################################
# functions to simulate genotype and phenotype data
######################################################

function simGeno( n::Int64, m::Int64, theta::Float64, p::Float64=0.5 )

    g = zeros(n,m)
    d = Bernoulli(p)
    
    g[:,1] = rand(d,n)

    d = Bernoulli(theta)
    
    for i=2:m
        r = rand(d,n)
        g[:,i] = r.*(1-g[:,i-1]) + (1-r).*g[:,i-1]
    end

    return g
end

function simPheno( n::Int64, beta::Array{Float64,2},
                   q::Array{Float64,2}, sigma2::Float64=1.0 )

    sigma = sqrt(sigma2)

    y = q*beta + sigma*randn(n)

    return y
end    

                   
