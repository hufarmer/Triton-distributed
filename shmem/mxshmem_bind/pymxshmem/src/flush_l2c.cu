#include <cuda.h>

__global__ void flush_l2c_dev() {
    asm ("wb_l2");
    asm ("arrive 0x40");//等待wb_l2执行完
    asm ("inv_l2");
}

extern "C" void flush_l2c(cudaStream_t stream) {
    flush_l2c_dev<<<1, 64, 0, stream>>>();
}