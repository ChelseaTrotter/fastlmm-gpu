###########################################
# genome scan functions
###########################################

#####################################################################
# we will estimate the variance components under the null first, and
# then perform weighted least squares across markers
#####################################################################

# genome scan for single phenotype, no covariates, 1-df genotypes;
# no missing data

function scan1( y::Array{Float64,2}, g::Array{Float64,2},
               K::Array{Float64,2}, reml::Bool=false )

    # number of individuals
    n = size(y,1)
    # add the intercept to the genotype matris
    g = [ones(n,1) g];
    # rotate the data with respect to the kinship matrix
    (yy,gg,lambda) = rotateData(y,g,K)
    # fit null model
    out0 = flmm(yy,gg[:,[1]],lambda,reml)

    w = weightCalc(out0.h2,lambda)

    # rescale by the weights
    yy = scale!(sqrt.(w),yy)
    gg = scale!(sqrt.(w),gg)

    nmar = size(g,2)-1
    lod = zeros(nmar)

    lod0 = wls(yy,gg[:,[1]]).ell
    
    for j=1:nmar
        lod[j] = wls(yy,gg[:,[1,j+1]]).ell
    end

    return lod - lod0
end

# weight function calculator

function weightCalc( h2::Float64, lambda::Array{Float64,1} )
    return 1.0./(h2*lambda+(1.0-h2))
end
