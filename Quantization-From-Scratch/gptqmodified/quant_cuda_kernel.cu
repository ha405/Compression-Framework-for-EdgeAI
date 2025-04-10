// quant_cuda_kernel.cu

#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
);

__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
);

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const  uint32_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   uint8_t* __restrict__ zeros,
    int height,
    int width
);


const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;
const int BLOCKWIDTH_8BIT = 128; // Example block width for 8bit kernel


void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data_ptr<scalar_t>(), mat.data_ptr<int>(), mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(), zeros.data_ptr<scalar_t>(),
        height, width
      );
    })
  );
}

void vecquant3matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<float>(),
    height, width
  );
}


void vecquant8matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  // mat shape: [infeatures / 4, outfeatures] -> height = infeatures / 4, width = outfeatures
  // vec shape: [infeatures]
  // mul shape: [outfeatures]
  // scales shape: [outfeatures]
  // zeros shape: [outfeatures] (uint8)
  int packed_rows = mat.size(0);
  int width = mat.size(1); // outfeatures
  int height = vec.size(0); // infeatures

  // Launch configuration maps threads to output columns (width)
  dim3 blocks((width + BLOCKWIDTH_8BIT - 1) / BLOCKWIDTH_8BIT);
  dim3 threads(BLOCKWIDTH_8BIT);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    vec.scalar_type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data_ptr<scalar_t>(),
        mat.data_ptr<uint32_t>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<uint8_t>(),
        height, // infeatures
        width   // outfeatures
      );
    })
  );
}


__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}


template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  if (threadIdx.x < BLOCKWIDTH) {
      int vec_idx = (row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x;
       if (vec_idx < height) { // Ensure reading within bounds of vec (height=infeatures)
          blockvec[threadIdx.x] = vec[vec_idx];
       } else {
           blockvec[threadIdx.x] = 0; // Pad with zero if out of bounds
       }
  }
  __syncthreads();

  if (col >= width) return; // Check column bounds

  scalar_t scale = scales[col];
  scalar_t zero = zeros[col]; // Float zero point for 3bit

  scalar_t res = 0;
  int i = width * row + col; // Index into mat [height=rows(packed), width=cols(outfeatures)]
  int k = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  // Assuming height is multiple of BLOCKHEIGHT for simplicity in loop structure
  // A more robust kernel handles arbitrary height/infeatures
  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width; // Move to the next packed row for this column
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

   __shared__ half2 blockvec[blockwidth2];
   if (threadIdx.x < blockwidth2) {
       int vec_idx = (row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x;
        // Assuming vec has length height = infeatures
       if (vec_idx * 2 + 1 < height) { // Check bounds for half2
           blockvec[threadIdx.x] = vec[vec_idx];
       } else if (vec_idx * 2 < height) {
           // Handle case where only the first half is valid
           // This requires careful handling or assuming vec length is even
            blockvec[threadIdx.x] = __halves2half2(((half*)vec)[vec_idx * 2], __float2half(0.0f));
       }
       else {
            blockvec[threadIdx.x] = __float2half2_rn(0.0f); // Pad with zero
       }
   }

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  __syncthreads();

  if (col >= width) return; // Check column bounds

  half2 scale = __float2half2_rn(scales[col]);
  half2 zero = __float2half2_rn(-zeros[col]); // Note: using -zeros[col]

  int i = width * row + col;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;


  while (k < blockwidth2) { // Iterate through input features (paired in half2)
    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[col], res);
}


template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,     // Input vector [infeatures]
    const  uint32_t* __restrict__ mat,     // Packed weights [infeatures / 4, outfeatures]
           scalar_t* __restrict__ mul,     // Output vector [outfeatures]
    const  scalar_t* __restrict__ scales,  // Scales [outfeatures]
    const   uint8_t* __restrict__ zeros,   // Integer zero points [outfeatures]
    int height,                           // Infeatures dimension
    int width                             // Outfeatures dimension
) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x; // Maps threads to output columns

    if (out_col >= width) return; // Ensure thread is within output bounds

    scalar_t scale = scales[out_col];
    uint8_t zero_point = zeros[out_col];
    scalar_t zero_point_f = static_cast<scalar_t>(zero_point);

    scalar_t accum = 0;

    // Iterate over the input features (height = infeatures)
    for (int k = 0; k < height; ++k) {
        // Calculate index into packed matrix
        int packed_row = k / 4;
        int sub_idx = k % 4; // Which 8-bit weight within the uint32 (0, 1, 2, or 3)

        // Read the packed uint32 value
        // Index: packed_row * width + out_col (Column-major access)
        uint32_t packed_val = mat[packed_row * width + out_col];

        // Unpack the specific 8-bit weight
        uint8_t q_val = (packed_val >> (sub_idx * 8)) & 0xFF;

        // Dequantize: W = scale * (Q - zero_point)
        scalar_t w_val = scale * (static_cast<scalar_t>(q_val) - zero_point_f);

        // Multiply and accumulate
        accum += w_val * vec[k];
    }

    // Atomically add the result for this output column
    // Assumes mul is initialized to zero before kernel launch
    atomicAdd(&mul[out_col], accum);
}