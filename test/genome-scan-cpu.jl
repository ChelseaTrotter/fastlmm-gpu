using Statistics



n = 100 #100 ~ 10k
m = 20000 #20k
r = 100 #100 ~ 1m



Y = rand(n, m)
G = rand(n, r)

#println("Y\n", Y)
#println("G\n", G)

#step 1: calculate standardized version of Y and G
@time Y_standard = (Y .- mean(Y)) / std(Y);
@time G_standard = (G .- mean(G)) / std(G);

#step 2: calculate R, matrix of corelation coefficients 
@time R = Y_standard' * G_standard;

#step 3: calculate proportion of variance explained 
@time R.*R; 

