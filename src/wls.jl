##################################################################
# wls: weighted least squares        
##################################################################        

import Base.inv

type Wls
    b::Array{Float64,2}
    sigma2::Float64
    ell::Float64
end
            
            
"""
wls: Weighted least squares estimation

y = outcome, matrix  
X = predictors, matrix  
w = weights (positive, inversely proportional to variance), one-dim vector

The variance estimate is maximum likelihood
"""

function wls(y::Array{Float64,2},X::Array{Float64,2},w::Array{Float64,1},
             reml::Bool=false,loglik=false)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)
    
    # check if weights are positive
    if(any(w.<=.0))
        error("Some weights are not positive.")
    end
         
    # square root of the weights
    sqrtw = sqrt.(w)
    # scale by weights
    # yy = y.*sqrtw
    yy = Diagonal(sqrtw)*y
    # XX = diagm(sqrtw)*X
    XX = Diagonal(sqrtw)*X
        
    # QR decomposition of the transformed data
    (q,r) = qr(XX)
    b = r\At_mul_B(q,yy)
    # estimate yy and calculate rss
    yyhat = XX*b
    # yyhat = q*At_mul_B(q,yy)
    rss = norm((yy-yyhat))^2

    if( reml )        
        sigma2 = rss/(n-p)
    else
        sigma2 = rss/n
    end

    # return coefficient and variance estimate
    logdetSigma = n*log(sigma2) - sum(log.(w)) 
    ell = -0.5 * ( logdetSigma + rss/sigma2 ) 
    if ( reml )
        ell -=  log(abs(det(r))) - (p/2)*(log(sigma2))
    end
        
    return Wls(b,sigma2,ell)
        
end

        
function wls(y::CuArray{Float64,2},X::CuArray{Float64,2},w::CuArray{Float64,1},
             reml::Bool=false,loglik=false)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)
    
    # check if weights are positive
    if(any(w.<=.0))
        error("Some weights are not positive.")
    end
         
    # square root of the weights
    sqrtw = sqrt.(w)
    # scale by weights
    # yy = y.*sqrtw
    yy = sqrtw.*y
    # XX = diagm(sqrtw)*X
    XX = CuArray(Diagonal(sqrtw))*X
        
    # solve
    # XXtXX = XX'*XX
    (q,r) = qr(XX)
    b = inv(r)*At_mul_B(q,yy)
    # estimate yy and calculate rss
    yyhat = XX*b
    # yyhat = q*At_mul_B(q,yy)
    rss = norm((yy-yyhat))^2

    if( reml )        
        sigma2 = rss/(n-p)
    else
        sigma2 = rss/n
    end

    # return coefficient and variance estimate
    logdetSigma = n*log(sigma2) - sum(CUDAnative.log.(w)) 
    ell = -0.5 * ( logdetSigma + rss/sigma2 ) 
    if ( reml )
        ell -=  0.5*log(abs(det(r))) - (p/2)*(log(sigma2))
    end
        
    return Wls(b,sigma2,ell)
        
end


function inv(x::CuArray)
    return CuArrays.BLAS.matinv_batched([x])[2][1]
end


        using CuArrays
        n = 1000;
        p = 10;
        b = ones(p,1);
        X = randn(n*2,p);
        Y = X*b+ randn(n*2,1);
        W = repeat([4.0; 1.0],inner=n);
        x = CuArray(X);
        y = CuArray(Y);
        w = CuArray(W);

        tic
