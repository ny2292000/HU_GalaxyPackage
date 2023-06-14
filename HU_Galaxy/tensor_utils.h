#pragma once

#include <torch/torch.h>
#include <vector>

inline std::vector<std::vector<double>> tensor_to_vec_of_vec(const torch::Tensor& tensor){
    std::vector<std::vector<double>> vec_of_vec;

    // Assumption: tensor is 2D
    int rows = tensor.size(0);
    int cols = tensor.size(1);

    vec_of_vec.resize(rows, std::vector<double>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            try {
                vec_of_vec[i][j] = tensor[i][j].item<double>();
            }catch (const char* e) {
                std::cout << e << std::endl;
            }
        }
    }
    return vec_of_vec;
}
