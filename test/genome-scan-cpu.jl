
# Date: Sep 17, 2018
# Authur: Chelsea Trotter
# This program tests how long does genome scan takes. 
# Currently I am testing only square matrix, (r >= m >= n). In the future, I may test whether the shape of the matrix affects computation time.



include("env.jl")
include("../src/benchmark.jl")

#n: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 
#m:                                            25600, 
#r:                                                , 51200, 102400, 204800, 409600, 819200, 1638400

function get_standardized_matrix(m)
    return (m .- mean(m)) ./ std(m)
end

function calculate_r(a::Array,b::Array)
    return LinAlg.BLAS.gemm('T', 'N', a,b)
    # return a' * b
end

function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.BLAS.gemm('T', 'N', a,b)
end


function my_isapprox(x,y)
    return isapprox(x,y, atol=1e-7)
end

function check_correctness(a, b)
    if(all(map(my_isapprox, a, b)))
        return "true"
    else
        return "false"
    end
end

function cpurun(a::Array, b::Array)
    #step 1: calculate standardized version of Y and G
    a_standard = get_standardized_matrix(a)
    b_standard = get_standardized_matrix(b)
    #step 2: calculate R, matrix of corelation coefficients
    r = calculate_r(a,b)
    #step 3: calculate proportion of variance explained 
    return r.*r
end

function gpurun(a::Array, b::Array)

    a_standard = get_standardized_matrix(a)
    b_standard = get_standardized_matrix(b)

    d_a = CuArray(a);
    d_b = CuArray(b);
    r = collect(calculate_r(d_a,d_b));
    return r.*r
end




n_max = 12800
m_max = 25600
# Matrix size of less than 1600 is very fast, basically have no comparison value to the profiling. But they are kept in here since that is what actual data may look like. 
matrix_size_range = [#=100, 200, 400, 800,1600,3200,=# 6400#=,12800,25600, 51200, 102400, 204800, 409600, 819200, 1638400=#]

dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
host = gethostname()

file = open("./timing/genome-scan-timing@$host@$dt_now.csv", "w")

for i in matrix_size_range

    n = 640 
    m = 640
    r = 640
    
    if(n > n_max)
        n = n_max
    end
    if(m > m_max)
        m = m_max
    end

    file = open("./timing/genome-scan-timing@$host@$dt_now.csv", "a")

    println("*************************** n: $n,m: $m, r: $r******************************")
    
    srand(123);

    Y = rand(n, m)
    G = rand(n, r)

    a_std = get_standardized_matrix(Y);
    b_std = get_standardized_matrix(G);

    cpu_result = benchmark(10, cpurun, a_std, b_std)
    gpu_result = benchmark(10, gpurun, a_std, b_std)
    speedup = cpu_result[3]/gpu_result[3]

    println("$m, $n, $r, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    write(file, "$m, $n, $r, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    close(file)

end

close(file)






