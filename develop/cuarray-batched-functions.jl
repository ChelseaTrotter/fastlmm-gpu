# helper function to get a device array of device pointers
function device_batch(batch::Array{T}) where {T<:CuArray}
  E = eltype(T)
  ptrs = [Base.unsafe_convert(Ptr{E}, arr.buf) for arr in batch]
  CuArray(ptrs)
end

#potrf batched function
for (fname, elty) in
    ((:cusolverDnSpotrfBatched,:Float32),
     (:cusolverDnDpotrfBatched,:Float64),
     (:cusolverDnCpotrfBatched,:Complex64),
     (:cusolverDnZpotrfBatched,:Complex128))
     @eval begin
         # cusolverStatus_t cusolverDnSpotrfBatched(
         #    cusolverDnHandle_t handle,
         #    cublasFillMode_t uplo,
         #    int n,
         #    float *Aarray[],
         #    int lda,
         #    int *infoArray,
         #    int batchSize);
         function potrf_batched!(uplo::BlasChar,
                                 Aarray::Array{CuMatrix{$elty},1})
            println("***** in potrf batched function! *****")
            for As in Aarray
                m,n = size(As)
                if m != n
                    throw(DimensionMismatch("All matrices must be square!"))
                end
            end
            m,n = size(Aarray[1])
            lda = max(1, stride(Aarray[1],2))
            Aptrs = device_batch(Aarray)
            info = CuArray{Cint}(length(Aarray))
            @check ccall(($(string(fname)), libcusolver),
                          cusolverStatus_t,
                          (cusolverDnHandle_t, cublasFillMode_t, Cint, Array{Ptr{$elty}}, Cint, Ptr{Cint}, Cint),
                          libcusolver_handle_dense[], uplo, n, Aptrs, lda, info, length(A))
            Aarray, info
        end
    end
end


function gemm_batched!(transA::BlasChar,
                       transB::BlasChar,
                       alpha::Float64,
                       A::Array{CuMatrix{Float64},1},
                       B::Array{CuMatrix{Float64},1},
                       beta::Float64,
                       C::Array{CuMatrix{Float64},1})
    if(length(A) != length(B) || length(A) != length(C))
        throw(DimensionMismatch(""))
    end
    #checking if matrix demention matches or not. TODO: is this for loop processed by GPU?
    #also, this does not check if every As has the same dimention.
    #It only checks if As, Bs, Cs has the same demention, which is ok until the next step, it uses A[1] as the leading dimention for all A[1...]
    for(As, Bs, Cs) in zip(A,B,C)
        m = size(As, transA == 'N' ? 1:2)
        k = size(As, transA == 'N' ? 2:1)
        n = size(Bs, transB == 'N' ? 2:1)
        if m!= size(Cs, 1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1:2)
            throw(DimensionMismatch(""))
        end
    end
    #Using A[1] as the leading dimention for all A[1...]. It should be true, but it won't throw an error until you encounter.
    m = size(A[1], transA == 'N' ? 1:2)
    k = size(A[1], transA == 'N' ? 2:1)
    n = size(B[1], transB == 'N' ? 2:1)

    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    lda = max(1,stride(A[1],2))
    ldb = max(1,stride(B[1],2))
    ldc = max(1,stride(C[1],2))
    Aptrs = device_batch(A)
    Bptrs = device_batch(B)
    Cptrs = device_batch(C)

    @check ccall(("gemm_batched!", libcublas), cublasStatus_t,
                (cublasHandle_t, cublasOperation_t,
                cublasOperation_t, Cint, Cint, Cint, Ptr{$elty},
                Ptr{Ptr{$elty}}, Cint, Ptr{Ptr{$elty}}, Cint, Ptr{$elty},
                Ptr{Ptr{$elty}}, Cint, Cint),
               libcublas_handle[], cutransA,
               cutransB, m, n, k, [alpha], Aptrs, lda, Bptrs, ldb, [beta],
               Cptrs, ldc, length(A))
  C
end



#=
for (fname, elty) in
        ((:cublasDgemmBatched,:Float64),
         (:cublasSgemmBatched,:Float32),
         (:cublasZgemmBatched,:Complex128),
         (:cublasCgemmBatched,:Complex64))
    @eval begin
        # cublasStatus_t cublasDgemmBatched(
        #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        #   int m, int n, int k,
        #   const double *alpha, const double **A, int lda,
        #   const double **B, int ldb, const double *beta,
        #   double **C, int ldc, int batchCount)
        function gemm_batched!(transA::BlasChar,
                               transB::BlasChar,
                               alpha::($elty),
                               A::Array{CuMatrix{$elty},1},
                               B::Array{CuMatrix{$elty},1},
                               beta::($elty),
                               C::Array{CuMatrix{$elty},1})
            if( length(A) != length(B) || length(A) != length(C) )
                throw(DimensionMismatch(""))
            end
            for (As,Bs,Cs) in zip(A,B,C)
                m = size(As, transA == 'N' ? 1 : 2)
                k = size(As, transA == 'N' ? 2 : 1)
                n = size(Bs, transB == 'N' ? 2 : 1)
                if m != size(Cs,1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end
            m = size(A[1], transA == 'N' ? 1 : 2)
            k = size(A[1], transA == 'N' ? 2 : 1)
            n = size(B[1], transB == 'N' ? 2 : 1)
            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = device_batch(A)
            Bptrs = device_batch(B)
            Cptrs = device_batch(C)
            @check ccall(($(string(fname)),libcublas), cublasStatus_t,
                         (cublasHandle_t, cublasOperation_t,
                          cublasOperation_t, Cint, Cint, Cint, Ptr{$elty},
                          Ptr{Ptr{$elty}}, Cint, Ptr{Ptr{$elty}}, Cint, Ptr{$elty},
                          Ptr{Ptr{$elty}}, Cint, Cint),
                         libcublas_handle[], cutransA,
                         cutransB, m, n, k, [alpha], Aptrs, lda, Bptrs, ldb, [beta],
                         Cptrs, ldc, length(A))
            C
        end
        function gemm_batched(transA::BlasChar,
                      transB::BlasChar,
                      alpha::($elty),
                      A::Array{CuMatrix{$elty},1},
                      B::Array{CuMatrix{$elty},1})
            C = CuMatrix{$elty}[similar( B[1], $elty, (size(A[1], transA == 'N' ? 1 : 2),size(B[1], transB == 'N' ? 2 : 1))) for i in 1:length(A)]
            gemm_batched!(transA, transB, alpha, A, B, zero($elty), C )
        end
        function gemm_batched(transA::BlasChar,
                      transB::BlasChar,
                      A::Array{CuMatrix{$elty},1},
                      B::Array{CuMatrix{$elty},1})
            gemm_batched(transA, transB, one($elty), A, B)
        end
    end
end
=#
