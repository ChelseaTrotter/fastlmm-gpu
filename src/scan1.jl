###########################################
# genome scan functions
###########################################

##############################################################
# (unweighted) linear regression genome scan with no covariates
##############################################################

function scan1( y::Array{Float64,2}, g::Array{Float64,2}, gpu::Bool )
               
    # number of individuals
    n = size(y,1)
    m = size(y,2)
    if(m>1) 
        error("Too many phenotypes.")
    end
        
    # the intercept 
    intrcpt = ones(n,1)

    # fit null model
    out0 = ls(y,intrcpt)

    # number of markers
    nmar = size(g,2)
    lod = zeros(nmar)

    # null model lod
    lod0 = out0.ell

    # orthogonalize phenotype and markers beforehand
    y0 = y-mean(y)
    g0 = copy(g)
        
    for j=1:nmar
        g0[:,j] = g0[:,j] - mean(g0[:,j])
    end

    # scan markers
    if(gpu)
        Y0 = CuArray(y0)
        G0 = CuArray(g0)
        for j=1:nmar
            lod[j] = ls(Y0,G0[:,[j]]).ell
        end
    else            
        for j=1:nmar
            lod[j] = ls(y0,g0[:,[j]]).ell
        end
    end
    return lod - lod0
end

    
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



#######################

    n = 10000
    m = 1000
    g = randn(n,m);
    mq = div(m,2);
    y = randn(n,1) + g[:,mq];

    # Y=CuArray(y);
    # G=CuArray(g);

    tic(); lod = scan1(y,g,true); toc()
    tic(); lod = scan1(y,g,false); toc()        
    # tic(); LOD = scan1(Y,G); toc()


