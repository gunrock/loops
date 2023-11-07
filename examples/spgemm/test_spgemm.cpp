#include <fstream>
#include <iostream>
#include <loops/algorithms/spgemm/thread_mapped.cuh>

using type_t = float;

void copyDeviceMtxToHost(const loops::matrix_t<type_t, loops::memory_space_t::device>& d_C, loops::matrix_t<type_t, loops::memory_space_t::host>& h_C){
    // Ensure the host matrix has the correct dimensions
    h_C.rows = d_C.rows;
    h_C.cols = d_C.cols;

    // Allocate memory for the host matrix data
    h_C.m_data.resize(d_C.rows * d_C.cols);

    // Copy matrix data from device to host
    cudaMemcpy(h_C.m_data.data(), d_C.m_data_ptr, sizeof(type_t) * d_C.rows * d_C.cols, cudaMemcpyDeviceToHost);

    // Update m_data_ptr on the host-side matrix_t object
    h_C.m_data_ptr = h_C.m_data.data();
}

void writeMtxToFile(loops::matrix_t<type_t, loops::memory_space_t::host>& C_host, int rows, int cols, const std::string& filename) {
    std::cout<<"filename: "<<filename<<std::endl;
    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outputFile << C_host(i, j);
            if (j < cols - 1) {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }
    outputFile.close();
}
