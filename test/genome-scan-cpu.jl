
# Date: Sep 17, 2018
# Authur: Chelsea Trotter
# This program tests how long does genome scan takes. 
# Currently I am testing only square matrix, (r >= m >= n). In the future, I may test whether the shape of the matrix affects computation time.






using Statistics
using Profile 

#using BenchmarkTools 
# temporaryly giving up on BenchmarkTools package, because it requires me to make modification to the code.
# eg:  Y_standard = (Y .- mean(Y)) / std(Y); 
# will change to @btime Y_standard = ($Y .- mean($Y)) / std($Y);
# This is because Y is a global variable, and you must add $ (interpolation) to avoid the error. 


#n: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 
#m:                                            25600, 
#r:                                                , 51200, 102400, 204800, 409600, 819200, 1638400




n_max = 12800
m_max = 25600

matrix_size_range = [100, 200, 400, 800, 1600#=, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600, 819200, 1638400=#]




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

    println("$n, $m, $r")
    
    Y = rand(n, m)
    G = rand(n, r)

    # println("Y\n", Y)
    # println("G\n", G)

    #run all functions once to prepare for profiling
    #step 1: calculate standardized version of Y and G
    Y_standard = (Y .- mean(Y)) / std(Y);
    G_standard = (G .- mean(G)) / std(G);

    #step 2: calculate R, matrix of corelation coefficients 
    R = Y_standard' * G_standard;

    #step 3: calculate proportion of variance explained 
    R.*R;


    #run all functions a second time for profiling. 
    #step 1: calculate standardized version of Y and G

    println("=======Get Standardized Y Matrix ======")
    @profile Y_standard = (Y .- mean(Y)) / std(Y);
    Profile.print()
    Profile.clear()

    println("=======Get Standardized G Matrix ======")
    @profile G_standard = (G .- mean(G)) / std(G);
    Profile.print()
    Profile.clear()

    #step 2: calculate R, matrix of corelation coefficients 
    println("=======Calculate R ======")
    @profile R = Y_standard' * G_standard;
    Profile.print()
    Profile.clear()

    #step 3: calculate proportion of variance explained 
    println("=======Calculate proportion of variance explained ======")
    @profile R.*R; 
    Profile.print()
    Profile.clear()



end










