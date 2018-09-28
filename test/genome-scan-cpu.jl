
# Date: Sep 17, 2018
# Authur: Chelsea Trotter
# This program tests how long does genome scan takes. 
# Currently I am testing only square matrix, (r >= m >= n). In the future, I may test whether the shape of the matrix affects computation time.



include("env.jl")

#n: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 
#m:                                            25600, 
#r:                                                , 51200, 102400, 204800, 409600, 819200, 1638400

function get_standardized_matrix(m)
    return (m .- mean(m)) ./ std(m)
end

function calculate_r(a::Array,b::Array)
    return a' * b
end

function calculate_r(a::CuArray,b::CuArray)
    return collect(CuArrays.BLAS.gemm('T', 'N', a,b))
end

function my_isapprox(x,y)
    return isapprox(x,y, atol=1e-5)
end

function check_correctness(a, b)
    if(all(map(my_isapprox, a, b)))
        return "true"
    else
        return "false"
    end
end

function myrun(a, b)

    a_standard = get_standardized_matrix(a)
    b_standard = get_standardized_matrix(b)
    r = calculate_r(a,b)
    return r.*r
end




n_max = 12800
m_max = 25600
# Matrix size of less than 1600 is very fast, basically have no comparison value to the profiling. But they are kept in here since that is what actual data may look like. 
matrix_size_range = [#=100, 200, 400, 800, 1600,=# 3200#=, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400=#]




for i in matrix_size_range
    n = i 
    m = i
    r = i
 
    if(n > n_max)
        n = n_max
    end
    if(m > m_max)
        m = m_max
    end

    println("*************************** $n, $m, $r******************************")
    
    Y = rand(n, m)
    G = rand(n, r)

    # run(Y,G)
    # Profile.clear()
    # @profile run(Y,G)
    # ProfileView.view()

    # println("Y\n", Y)
    # println("G\n", G)

    #step 1: calculate standardized version of Y and G
    Y_standard = get_standardized_matrix(Y);
    G_standard = get_standardized_matrix(G);

    d_y = CuArray(Y_standard);
    d_g = CuArray(G_standard);

    #step 2: calculate R, matrix of corelation coefficients 
    R1 = calculate_r(Y_standard, G_standard);
    R2 = calculate_r(d_y, d_g);

    #step 3: calculate proportion of variance explained 
    r1_result = R1.*R1;
    r2_result = R2.*R2;

    println("correct? :" , my_isapprox(r1_result,r2_result))

    #time it

    #step 2: calculate R, matrix of corelation coefficients 
    println("*** hello ***")
    t1 = @belapsed r1 = calculate_r($Y_standard,$G_standard);;
    println("====== time of cpu: ", t1);
    t2 = @belapsed R2 = calculate_r($d_y, $d_g);
    println("====== time of gpu: ", t2);


    #=
    #run all functions a second time for profiling. 
    #step 1: calculate standardized version of Y and G
    println("=======Get Standardized Y Matrix ======")
    # Profile.clear()
    @profile Y_standard = get_standardized_matrix(Y);

    Profile.print()   
    

    println("=======Get Standardized G Matrix ======")
    Profile.clear()
    @profile G_standard = get_standardized_matrix(G);
    Profile.print()
    
 
    #step 2: calculate R, matrix of corelation coefficients 
    println("=======Calculate R ======")
    Profile.clear()
    @profile R = calculate_r(Y_standard, G_standard);
    Profile.print(C = true)
    
 
    #step 3: calculate proportion of variance explained 
    println("=======Calculate proportion of variance explained ======")
    R.*R;
    Profile.clear()
    @profile R.*R; 
    Profile.print()
    
    =#


end







