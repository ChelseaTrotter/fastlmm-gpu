
# Date: Sep 17, 2018
# Authur: Chelsea Trotter
# This program tests how long does genome scan takes. 
# Currently I am testing only square matrix, (r >= m >= n). In the future, I may test whether the shape of the matrix affects computation time.



include("env.jl")
include("../src/benchmark.jl")
import CuArrays.CuArray
import Base.@elapsed 



#n: 100, 200, 400, 800, 1600, 3200, 6400, 12800, 
#m:                                            25600, 
#r:                                                , 51200, 102400, 204800, 409600, 819200, 1638400

function get_standardized_matrix(m)
    return (m .- mean(m)) ./ std(m);
end

function calculate_r(a::Array,b::Array)
    return LinAlg.BLAS.gemm('T', 'N', a,b);
    # return a' * b
end

function calculate_r(a::CuArray,b::CuArray)
    return CuArrays.CUBLAS.gemm('T', 'N', a,b);

end   



function my_isapprox(x,y)
    return isapprox(x,y, atol=1e-7);
end

function check_correctness(a, b)
    if(all(map(my_isapprox, a, b)))
        return "true"
    else
        return "false"
    end
end

# function my_kernel()
# #step 1: calculate standardized version of Y and G ( mean => m - mean => m - mean / standard deviation of m)


# #step 2: calculate R, matrix of corelation coefficients (gemm)

# #step 3: calculate proportion of variance explained (square every entry )

# end

function cpurun(a::Array, b::Array)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    #step 2: calculate R, matrix of corelation coefficients
    r = calculate_r(a_std,b_std);
    #step 3: calculate proportion of variance explained 
    return r.*r;
end

function gpurun(a::Array, b::Array)
    a_std = get_standardized_matrix(a);
    b_std = get_standardized_matrix(b);
    d_a = CuArray(a_std);
    d_b = CuArray(b_std);
    d_r = calculate_r(d_a,d_b);
    #Get total number of threads 
    ndrange = prod(size(d_r))
    #Get maximum number of threads per block
    dev = device()
    threads = attribute(dev, CUDAdrv.WARP_SIZE)
    blocks = min(ceil(ndrange/threads), attribute(dev, CUDAdrv.MAX_GRID_DIM_X))
    @cuda blocks=blocks threads=threads square_kernel(d_r)
    return Array(d_r)
end

function square_kernel(a) 
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i < ndrange+1)
        a[i] = a[i] * a[i]
    end
    return 
end

n_max = 12800;
m_max = 25600;

# Matrix size of less than 1600 is very fast, basically have no comparison value to the profiling. But they are kept in here since that is what actual data may look like. 
matrix_size_range = [100#=, 200, 400, 800,1600,3200,6400,12800,25600, 51200, 102400, 204800, 409600, 819200, 1638400=#]

dt_now = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS");
host = gethostname();

# file = open("./timing/genome-scan-timing@$host@$dt_now.csv", "w");

for n in matrix_size_range
    m = n
    r = n
    if(n > n_max)
        n = n_max;
    end
    if(m > m_max)
        m = m_max;
    end

    # file = open("./timing/genome-scan-timing@$host@$dt_now.csv", "a");

    println("*************************** n: $n,m: $m, r: $r******************************");
    
    srand(123);

    Y = rand(n, m);
    G = rand(n, r);

    #step 1: calculate standardized version of Y and G

    std_y_time = @elapsed y_std = get_standardized_matrix(Y);
    std_g_time = @elapsed g_std = get_standardized_matrix(G);
    cpu_r_time = @elapsed r1 = calculate_r(y_std,g_std)

    d_y = CuArray(y_std);
    d_g = CuArray(g_std);
    gpu_r_time = @elapsed r2 = collect(calculate_r(d_y,d_g));
    sq_time = @elapsed cpu_r_sq = r1 * r1 

    total_time = std_y_time + std_g_time + cpu_r_time + sq_time


    println("Getting standard Y: $std_y_time seconds $((std_y_time/total_time)*100) %
             Getting standard G: $std_g_time seconds $((std_g_time/total_time)*100) %
             CPU gemm: $cpu_r_time seconds           $((cpu_r_time/total_time)*100) %
             GPU gemm: $gpu_r_time seconds           $(gpu_r_time/cpu_r_time)  speedup ratio
             R square: $sq_time seconds              $((sq_time/total_time)*100) % ")

    

    cpu_result = benchmark(1, cpurun, Y, G);
    gpu_result = benchmark(1, gpurun, Y, G);
    speedup = cpu_result[3]/gpu_result[3];

    println("$m, $n, $r, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    # println("std_time for Y is $std_time")
    # write(file, "$m, $n, $r, $(cpu_result[3]),  $(gpu_result[3]), $speedup\n");
    # close(file);

end

# close(file)






