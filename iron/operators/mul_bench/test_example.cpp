#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include <vector>
#include <cmath>
#include <random>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN 
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using H_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X


void linear(
    const std::vector<std::vector<DTYPE_IN>>& I,
    const std::vector<std::vector<DTYPE_IN>>& W,
    std::vector<std::vector<DTYPE_OUT>>& C
)
{

    size_t M = I.size(); // Number of time steps t
    size_t K = I[0].size(); // Input dimension d
    size_t N = W[0].size(); // Output dimension h

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float ref = 0;
            for(size_t k = 0; k < K; ++k){
                ref += static_cast<DTYPE_ACC>(I[i][k]) * static_cast<DTYPE_ACC>(W[k][j]);
            }
            C[i][j] = static_cast<DTYPE_OUT>(ref);
        }
    }
}

void tanh_vector(
    std::vector<std::vector<DTYPE_OUT>>& I
   // const std::vector<std::vector<DTYPE_OUT>>& O
)
{
    size_t M = I.size(); // Number of time steps t
    size_t N = I[0].size(); // Output dimension h
    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float res = std::tanh(static_cast<float>(I[i][j]));
            I[i][j] = static_cast<DTYPE_OUT>(res);
        }
    }
}

void sigmoid_vector(
    std::vector<std::vector<DTYPE_OUT>>& I
    //const std::vector<std::vector<DTYPE_OUT>>& O
)
{
    // using tanh approximation
    size_t M = I.size(); // Number of time steps t
    size_t N = I[0].size(); // Output dimension h
    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float res = std::tanh(static_cast<float>(I[i][j])*0.5);
            res = 0.5*(1+res);
            I[i][j] = static_cast<DTYPE_OUT>(res);
        }
    }
}

void ewise_mult(
    const std::vector<std::vector<DTYPE_OUT>>& A,
    const std::vector<std::vector<DTYPE_OUT>>& B,
    std::vector<std::vector<DTYPE_OUT>>& C
)
{

    size_t M = A.size(); // Number of time steps t
    size_t N = A[0].size(); // Output dimension h

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            C[i][j] =static_cast<DTYPE_OUT>(static_cast<DTYPE_ACC>(A[i][j]) * static_cast<DTYPE_ACC>(B[i][j]));
        }
    }
}


void ewise_add(
    const std::vector<std::vector<DTYPE_OUT>>& A,
    const std::vector<std::vector<DTYPE_OUT>>& B,
    std::vector<std::vector<DTYPE_OUT>>& C
)
{

    size_t M = A.size(); // Number of time steps t
    size_t N = A[0].size(); // Output dimension h

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            C[i][j] =static_cast<DTYPE_OUT>(static_cast<DTYPE_ACC>(A[i][j]) + static_cast<DTYPE_ACC>(B[i][j]));
        }
    }
}

void ewise_mulmin(
    const std::vector<std::vector<DTYPE_OUT>>& A,
    const std::vector<std::vector<DTYPE_OUT>>& B,
    std::vector<std::vector<DTYPE_OUT>>& C
)
{

    size_t M = A.size(); // Number of time steps t
    size_t N = A[0].size(); // Output dimension h

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            C[i][j] =static_cast<DTYPE_OUT>((1-static_cast<DTYPE_ACC>(A[i][j])) * static_cast<DTYPE_ACC>(B[i][j]));
        }
    }
}


std::random_device rd;
std::mt19937 gen (rd());
std::uniform_real_distribution<float> dist(-0.02, 0.02);

void random_vector(
    std::vector<std::vector<DTYPE_IN>>& I
)
{
    size_t M = I.size(); // Number of time steps t
    size_t N = I[0].size(); // Output dimension h
    //std::random_device rd;
    //std::mt19937 gen (rd());
    //std::uniform_real_distribution<float> dist(-0.02, 0.02);
    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float num = dist(gen);
            I[i][j] = (DTYPE_IN) num; //static_cast<DTYPE_IN>(num);
        }
    }
}


int verify(
    const std::vector<std::vector<DTYPE_IN>>& X,
    const std::vector<std::vector<DTYPE_IN>>& H,
    const std::vector<std::vector<DTYPE_IN>>& Wir,
    const std::vector<std::vector<DTYPE_IN>>& Wiz,
    const std::vector<std::vector<DTYPE_IN>>& Win,
    const std::vector<std::vector<DTYPE_IN>>& Whr,
    const std::vector<std::vector<DTYPE_IN>>& Whz,
    const std::vector<std::vector<DTYPE_IN>>& Whn,
    const std::vector<std::vector<DTYPE_OUT>>& C,
    int verbosity = 0,
    float tolerance = 0.165f // absolute percent difference
){
    int errors = 0;
    size_t M = X.size(); // Number of time steps t
    size_t K = X[0].size(); // Input dimension d
    size_t N = Wir[0].size(); // Output dimension h

    // linear operations
    std::vector<std::vector<DTYPE_OUT>> GA(M, std::vector<DTYPE_OUT>(N));
    std::vector<std::vector<DTYPE_OUT>> GB(M, std::vector<DTYPE_OUT>(N));
    std::vector<std::vector<DTYPE_OUT>> GC(M, std::vector<DTYPE_OUT>(N));
    std::vector<std::vector<DTYPE_OUT>> GD(M, std::vector<DTYPE_OUT>(N));
    std::vector<std::vector<DTYPE_OUT>> GE(M, std::vector<DTYPE_OUT>(N));
    std::vector<std::vector<DTYPE_OUT>> GF(M, std::vector<DTYPE_OUT>(N));

    linear(X, Wir, GA);
    linear(H, Whr, GB);
    linear(H, Whn, GC);
    linear(X, Win, GD);
    linear(X, Wiz, GE);
    linear(H, Whz, GF);

    // the rest lol
    std::vector<std::vector<DTYPE_OUT>> GR(M, std::vector<DTYPE_OUT>(N));
    ewise_add(GA, GB, GR);
    sigmoid_vector(GR);
    std::vector<std::vector<DTYPE_OUT>> GG(M, std::vector<DTYPE_OUT>(N));
    ewise_mult(GR, GC, GG);
    std::vector<std::vector<DTYPE_OUT>> GN(M, std::vector<DTYPE_OUT>(N));
    ewise_add(GD, GG, GN);
    tanh_vector(GN);
    std::vector<std::vector<DTYPE_OUT>> GZ(M, std::vector<DTYPE_OUT>(N));
    ewise_add(GE, GF, GZ);
    sigmoid_vector(GZ);
    std::vector<std::vector<DTYPE_OUT>> GJ(M, std::vector<DTYPE_OUT>(N));
    ewise_mult(GZ, H, GJ);
    //ewise_mult(GZ, GZ, GJ);
    std::vector<std::vector<DTYPE_OUT>> GK(M, std::vector<DTYPE_OUT>(N));
    ewise_mulmin(GZ, GN, GK);

    std::vector<std::vector<DTYPE_OUT>> GH(M, std::vector<DTYPE_OUT>(N));
    ewise_mulmin(GK, GJ, GH);

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float ref = static_cast<DTYPE_ACC>(GH[i][j]);
            float c_val = static_cast<DTYPE_ACC>(C[i][j]);
            float percent_diff = std::abs(ref - c_val)/ref;
            if(percent_diff > tolerance){
                std::cout << "Error at C[" << i << "][" << j << "]: "
                          << C[i][j] << "!=" << ref
                          << " from dot A[" << i << "], B[" << j << "] and H[" << j << "]\n";
                errors++;
            } 
            else if(verbosity >= 1){
                std::cout << "Correct output C[" << i << "][" << j << "]: "
                          << C[i][j] << " == " << ref << "\n";
            }
        }
    }
    return errors;
}




// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]){

    // ------------------------------------------------------
    // Parse program arguments
    // ------------------------------------------------------
    cxxopts::Options options("Linear Layer Test Using Matrix Multiplication");
    cxxopts::ParseResult vm;
    matmul_common::add_default_options(options);

    matmul_common::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();
    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();
    int trace_size = vm["trace_sz"].as<int>();

    // ------------------------------------------------------
    // Configure this to match your design's buffer size
    // ------------------------------------------------------
    int M = vm["rows"].as<int>();
    int K = vm["inner"].as<int>(); 
    int N = vm["columns"].as<int>();

    size_t input_size_x = M * K * 8 * sizeof(A_DATATYPE);
    size_t input_size_h = M * K * sizeof(A_DATATYPE);
    size_t weight_size_x = K * 8 * N * sizeof(B_DATATYPE);
    size_t weight_size_h = K * N * sizeof(B_DATATYPE);
    size_t output_size = M * N  * sizeof(C_DATATYPE);
    
    // Load instruction sequence
    std::vector<uint32_t> instr_v = 
        test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // ------------------------------------------------------
    // Get device, load the xclbin & kernel and register them
    // ------------------------------------------------------
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
    
    // Load the kernel
    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
    std::string Node = vm["kernel"].as<std::string>();

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                [Node, verbosity](xrt::xclbin::kernel &k) {
                                    auto name = k.get_name();
                                    if (verbosity >= 1){
                                        std::cout << "Name: " << name << std::endl;
                                    }
                                    return name.rfind(Node, 0) == 0;
                                });
    auto kernelName = xkernel.get_name();

    // Register xclbin
    if(verbosity >= 1)
        std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
                  << "\n";
    device.register_xclbin(xclbin);

    // Get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << "\n";
    auto kernel = xrt::kernel(context, kernelName);

    // ------------------------------------------------------
    // Initialize input/ output buffer sizes and sync them
    // ------------------------------------------------------
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inputs = 
        xrt::bo(device, (input_size_x + input_size_h), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

    auto bo_wr = 
        xrt::bo(device, (weight_size_x + weight_size_h), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    auto bo_wn = 
        xrt::bo(device, (weight_size_x + weight_size_h), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_wz = 
        xrt::bo(device, (weight_size_x + weight_size_h), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));


    auto bo_out = 
        xrt::bo(device, output_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));
    
    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Initialize instruction buffer
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    std::cout << "pre inputs\n";
    // Initialize input buffer -- holds both x and h
    A_DATATYPE *bufIn = bo_inputs.map<A_DATATYPE *>();
    // +1 is for the ones inserted for linear matmul
    std::vector<std::vector<A_DATATYPE>> XVec(M, std::vector<A_DATATYPE>(K*8));
    std::vector<std::vector<A_DATATYPE>> HVec(M, std::vector<A_DATATYPE>(K));

    random_vector(XVec);
    random_vector(HVec);
    // Flatten to buffer
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K*8; j++){
            size_t flat_ind = (i * K*8) + j;
            bufIn[flat_ind] = XVec[i][j];
        }
    }
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
            size_t flat_ind = (i * K) + j;
            bufIn[(M*(K*8))+flat_ind] = HVec[i][j];
        }
    }

    std::cout << "inputs done.\n";

    // Initialize wr  buffer
    B_DATATYPE *bufWr = bo_wr.map<B_DATATYPE *>();
    // weights + biases
    std::vector<std::vector<B_DATATYPE>> WirVec(K*8, std::vector<A_DATATYPE>(N));
    std::vector<std::vector<B_DATATYPE>> WhrVec(K, std::vector<A_DATATYPE>(N));

    random_vector(WirVec);
    random_vector(WhrVec);

    // Flatten to buffer
    for(int i = 0; i < K*8; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            //std::cout << flat_ind << std::endl;
            bufWr[flat_ind] = WirVec[i][j];
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            bufWr[((K*8)*N)+flat_ind] = WhrVec[i][j];
        }
    }

    std::cout << "wr done.\n";

    // Initialize wz  buffer
    B_DATATYPE *bufWz = bo_wz.map<B_DATATYPE *>();
    // weights + biases
    std::vector<std::vector<B_DATATYPE>> WizVec(K*8, std::vector<A_DATATYPE>(N));
    std::vector<std::vector<B_DATATYPE>> WhzVec(K, std::vector<A_DATATYPE>(N));

    random_vector(WizVec);
    random_vector(WhzVec);


    // Flatten to buffer
    for(int i = 0; i < K*8; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            bufWz[flat_ind] = WizVec[i][j];
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            bufWz[((K*8)*N)+flat_ind] = WhzVec[i][j];
        }
    }

    std::cout << "wz done.\n";

    // Initialize wn  buffer
    B_DATATYPE *bufWn = bo_wn.map<B_DATATYPE *>();
    // weights + biases
    std::vector<std::vector<B_DATATYPE>> WinVec(K*8, std::vector<A_DATATYPE>(N));
    std::vector<std::vector<B_DATATYPE>> WhnVec(K, std::vector<A_DATATYPE>(N));

    random_vector(WinVec);
    random_vector(WhnVec);

    // Flatten to buffer
    for(int i = 0; i < K*8; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            bufWn[flat_ind] = WinVec[i][j];
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            size_t flat_ind = (i * N) + j;
            bufWn[((K*8)*N)+flat_ind] = WhnVec[i][j];
        }
    }

    std::cout << "wn done.\n";

    // Initialize outY buffer
    char *bufOut = bo_out.map<char *>();
    memset(bufOut, 0, output_size);

    std::cout << "synchronizing.\n";
    // Sync buffers to update input buffer values
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_wr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_wz.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_wn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ------------------------------------------------------
    // Initialize run configs
    // ------------------------------------------------------
    unsigned num_iter = n_iterations + n_warmup_iterations;
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;

    int errors = 0;

    // ------------------------------------------------------
    // Main run loop
    // ------------------------------------------------------
    for (unsigned iter = 0; iter < num_iter; iter++){

        // Run kernel
        if (verbosity >= 1)
            std::cout << "Running Kernel.\n";
        auto start = std::chrono::high_resolution_clock::now();
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inputs, bo_wr, bo_wn, bo_wz, bo_out);
        run.wait();
        auto stop = std::chrono::high_resolution_clock::now();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if(iter < n_warmup_iterations){
            /* Warmup iterations do not count towards average runtime. */
            continue;
        }
        
        // Copy output results and verify they are correct
        C_DATATYPE* typedBuf = reinterpret_cast<C_DATATYPE*>(bufOut);
        // Create a 2D vector to store the result
        std::vector<std::vector<C_DATATYPE>> OutVec(M, std::vector<C_DATATYPE>(N));
        // Fill OutVec with the flattened output
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                size_t flat_index = i * N + j;  // Row-major layout
                OutVec[i][j] = typedBuf[flat_index];
            }
        }

        if(do_verify){
            if(verbosity >= 1){
                std::cout << "Verifying results ..." << std::endl;
            }
            auto vstart = std::chrono::system_clock::now();
            errors = verify(XVec, HVec, WirVec, WizVec, WinVec, WhrVec, WhzVec, WhnVec,OutVec); //verify(XVec, HVec, WXVec, WHVec, BXVec, BHVec, CVec, verbosity);
            auto vstop = std::chrono::system_clock::now();
            float vtime = 
                std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
                    .count();
            if (verbosity >= 1){
                std::cout << "Verify time: " << vtime << "secs." << std::endl;
            }
        }
        else{
            if(verbosity >= 1){
                std::cout << "WARNING: results not verified." << std::endl;
            }
        }

        // Write trace values if trace_size > 0
        if(trace_size > 0){
            test_utils::write_out_trace(((char *)bufOut) + output_size, trace_size, 
                                      vm["trace_file"].as<std::string>());
        }

        // Accumulate run times
        float npu_time =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();

        npu_time_total += npu_time;
        npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
        npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
    }

    // ------------------------------------------------------
    // Display results
    // ------------------------------------------------------

    // TODO - Mac count to guide gflops
    float macs = static_cast<float>(M * (K+1) * N * 2); //todo wrong
    std::cout << std::endl
              << "Avg NPU time: " << npu_time_total / n_iterations << "us."
              << std::endl;
    if (macs > 0)
        std::cout << "Avg NPU gflops: "
                  << macs / (1000 * npu_time_total / n_iterations) << std::endl;
    
    std::cout << std::endl 
              << "Min NPU time: " << npu_time_min << "us." << std::endl;
    if (macs > 0)
        std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
                  << std::endl;g will be in BA7180 tomorrow (usual room.
    
    std::cout << std::endl
              << "Max NPU time: " << npu_time_max << "us." << std::endl;
    if(macs > 0)
        std::cout << "Min NPU gflops: " << macs/ (1000 * npu_time_max)
                  << std::endl;

    if(!errors){
        std::cout << "\nPass!\n\n";
        return 0;
    }
    else{
        std::cout << "\nError count: " << errors << "\n\n";
        std::cout << "\nFailed.\n\n";
        return 1;
    }
}