#This file records all the dependencies that will be used. 
#In the begining of the project, all these dependencies will be checked whether they are already installed. 
#So please follow the format that starting of the line is "using", with 1 space, then following the package name. 
#Be very careful that after the package name, there should be no white space! 

function check_pkg_installation(envfile)
    # while there is next line in inputfile
    #     if line start with using 
    #         get the package name after using 
    #         if (package is not installed in Pkg.installed())
    #             install package. 
    #     end
    # end 

    file = open(envfile)
    while !eof(file)
        x = readline(file)
        if length(x) > 5 && x[1:5] == "using"
            println(x)
            package_name = x[7:end]
            !haskey(Pkg.installed(), package_name) && Pkg.add(package_name)
        end
    end
end

check_pkg_installation("env.jl")

using Statistics
# using Profile
using CuArrays
# using ProfileView
using BenchmarkTools

# using ProfileView # too many compilation error when installing package. giving up on this one. 


# eg:  Y_standard = (Y .- mean(Y)) / std(Y); 
# will change to @btime Y_standard = ($Y .- mean($Y)) / std($Y);
# This is because Y is a global variable, and you must add $ (interpolation) to avoid the error. 

