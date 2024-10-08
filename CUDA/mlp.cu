#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// CUDA based sigmoid function
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

__global__ void forward_layer(double* input, double* weights, double* bias, double* output,
                              int input_size, int hidden_size) {
    // Get the block size for the cuda
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Run on needed threads only
    if (idx < hidden_size) {
        output[idx] = bias[idx];
        for (int j = 0; j < input_size; j++) {
            // Note that weights is a 1-D vector
            output[idx] += input[j] * weights[idx * input_size + j];
        }
        output[idx] = sigmoid(output[idx]);
    }
}

__global__ void backward_layer_hidden(double* delta, double* delta_all, double* hidden, double* input, double* weights2,
                                      double* weights1, double* bias, int input_size, int output_size, int hidden_size,
                                      double lr) {
    // Get the block size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        double error = 0.0;
        // Get the error based on the layers
        for (int j = 0; j < output_size; j++) {
            error += delta[j] * weights2[idx * output_size + j];
        }

        // update the delta with the error and get new weights
        delta_all[idx] = error * sigmoid_derivative(hidden[idx]);
        bias[idx] += delta_all[idx] * lr;

        // Run the update equations
        for (int j = 0; j < input_size; j++) {
            weights1[j * hidden_size + idx] += lr * delta_all[idx] * input[j];
        }
    }
}

__global__ void backward_layer_output(double* output, double* target, double* hidden, double* weights2,
                                      double* bias2, double* delta_output, int output_size, int hidden_size,
                                      double lr) {
    // Get the block size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        double error = 0.0;
        // Get the error based on the activationsc
        error = target[idx] - output[idx];

        // update the delta with the error and get new weights
        delta_output[idx] = error * sigmoid_derivative(output[idx]);
        bias2[idx] += delta_output[idx] * lr;


        // Run the update equations
        for (int j = 0; j < hidden_size; j++) {
            weights2[j * output_size + idx] += lr * delta_output[idx] * hidden[j];

        }
    }
}

// Define the MLP structure
class MLP {
    // All the static shit
    int num_layers;
    std::vector<int> layer_sizes;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;

    // All the pointers
    double *d_input, *d_output, *d_target;
    std::vector<double*> d_layer_inputs;
    std::vector<double*> d_layer_output;
    std::vector<double*> d_deltas;
    std::vector<double*> d_weights;
    std::vector<double*> d_biases;
    
    public:
        void initialize_layer(int layer){
            srand(static_cast<unsigned>(time(0)));
            int input_size = this->layer_sizes[layer];
            int output_size = this->layer_sizes[layer + 1];

            // Loop and init each memeber
            for (int i = 0; i < input_size * output_size; i++) {
                this->weights[layer][i] = ((double)rand() / RAND_MAX) - 0.5;
            }
            for (int i = 0; i < output_size; i++) {
                this->biases[layer][i] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }

        MLP(const std::vector<int>& layer_sizes) {
            // Init the instance variable
            this->layer_sizes = layer_sizes;
            this->num_layers = layer_sizes.size();

            // Init the other members
            this->weights.resize(this->num_layers - 1);
            this->biases.resize(this->num_layers - 1);
            this->d_weights.resize(this->num_layers - 1);
            this->d_biases.resize(this->num_layers - 1);

            this->d_layer_inputs.resize(this->num_layers);
            this->d_layer_output.resize(this->num_layers);
            this->d_deltas.resize(this->num_layers);


            // Let's init all the members
            for (int i = 0; i < this->num_layers - 1; i++) {
                int in_layer_size = layer_sizes[i];
                int out_layer_size = layer_sizes[i + 1];
                this->weights[i].resize(in_layer_size * out_layer_size);
                this->biases[i].resize(out_layer_size);

                // Allocate the cuda memeory to all the pointers
                cudaMalloc(&d_weights[i], weights[i].size() * sizeof(double));
                cudaMalloc(&d_biases[i], biases[i].size() * sizeof(double));
                cudaMalloc(&d_layer_inputs[i], in_layer_size * sizeof(double));
                cudaMalloc(&d_layer_output[i], out_layer_size * sizeof(double));
                cudaMalloc(&d_deltas[i], out_layer_size * sizeof(double));

                // Init a layer that was just established
                initialize_layer(i);

                // Allocate space
                cudaMemcpy(d_weights[i], weights[i].data(), weights[i].size() * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_biases[i], biases[i].data(), biases[i].size() * sizeof(double), cudaMemcpyHostToDevice);
            }

            // Few mem allocation at the last
            int input_size = layer_sizes[0];
            int output_size = layer_sizes[num_layers - 1];
            cudaMalloc(&d_target, output_size * sizeof(double));
            cudaMalloc(&d_input, input_size * sizeof(double));
            cudaMalloc(&d_output, output_size * sizeof(double));
            cudaMalloc(&d_layer_inputs[num_layers - 1], output_size * sizeof(double));
            cudaMalloc(&d_layer_output[num_layers - 1], output_size * sizeof(double));
            cudaMalloc(&d_deltas[num_layers - 1], output_size * sizeof(double));
        }

        // Define the forward process CODE IS WORKING AS EXPECTED///
        void forward(const std::vector<double>& input, std::vector<double>& output) { 
            // Allocate the memory
            cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_layer_inputs[0], d_input, input.size() * sizeof(double), cudaMemcpyDeviceToDevice);

            // Loop and do forward pass
            for (int i = 0; i < num_layers - 1; i++) {
                int input_size = layer_sizes[i];
                int hidden_size = layer_sizes[i + 1];

                forward_layer<<<(hidden_size + 255) / 256, 256>>>(d_layer_inputs[i], d_weights[i], d_biases[i],
                                                                  d_layer_output[i], input_size, hidden_size);

                cudaDeviceSynchronize();
                cudaMemcpy(d_layer_inputs[i + 1], d_layer_output[i], hidden_size * sizeof(double), cudaMemcpyDeviceToDevice);
            }

            // Reize the outputss
            cudaMemcpy(d_output, d_layer_output[num_layers - 2], layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);
            output.resize(layer_sizes[num_layers - 1]);
            cudaMemcpy(output.data(), d_layer_output[num_layers - 2],
                       layer_sizes[num_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);
        }

        void backward(const std::vector<double>& targets, int lr) {
            // Allocate memory
            cudaMemcpy(d_target, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice);
        
            // Backprop the last layer first to get the output deltas
            backward_layer_output<<<(layer_sizes[num_layers - 1]) + 255 / 256, 256>>>(d_output, d_target, d_layer_inputs[num_layers - 2],
                                                                                      d_weights[num_layers - 2], d_biases[num_layers - 2],
                                                                                      d_deltas[num_layers - 1], layer_sizes[num_layers - 1],
                                                                                      layer_sizes[num_layers - 2], lr);
            cudaDeviceSynchronize();

            // Pass through the rest of the layers
            for (int i = num_layers - 1; i > 1; i--) {
                backward_layer_hidden<<<(layer_sizes[i - 1]) + 255 / 256, 256>>>(d_deltas[i - 1], d_deltas[i - 2],
                                                                                 d_layer_output[i - 1], d_layer_output[i - 2],
                                                                                 d_weights[i - 1], d_weights[i - 2],
                                                                                 d_biases[i - 1], layer_sizes[i - 2], layer_sizes[i],
                                                                                 layer_sizes[i - 1], lr);
                cudaDeviceSynchronize();
            }
        }

        ~MLP() {
            for (int i = 0; i < num_layers - 1; ++i) {
                cudaFree(d_weights[i]);
                cudaFree(d_biases[i]);
                cudaFree(d_layer_inputs[i]);
                cudaFree(d_layer_output[i]);
                cudaFree(d_deltas[i]);
            }
            cudaFree(d_input);
            cudaFree(d_output);
        }
};

int main() {
    std::cout << "Simple MLP functions......" << std::endl;
    std::cout << "Loading model" << std::endl;
    std::vector<int> layer_sizes = {7, 2, 1};
    std::cout << "Creating model" << std::endl;
    MLP mlp_mod(layer_sizes);

    // Make some dummy input
    std::vector<std::vector<double>> inputs = {
        {1, 0, 0, 1, 1, 1, 1},
        {0, 0, 0, 1, 1, 0, 1},
        {0, 1, 0, 0, 1, 1, 1},
        {0, 0, 0, 1, 0, 0, 1},
        {0, 0, 1, 0, 1, 1, 1},
        {0, 0, 0, 1, 1, 0, 1},
        {1, 0, 0, 1, 1, 1, 0},
    };

    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {1},
        {1},
        {1},
        {0}
    };

    double learningRate = 0.1;

    for (int epoch = 0; epoch < 100; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output;
            mlp_mod.forward(inputs[i], output);
            mlp_mod.backward(targets[i], learningRate);
        }
    }

    std::vector<double> output;
    for (const auto& input: inputs) {
        mlp_mod.forward(input, output);
        std::cout << "Input: ";
        for (double i: input) {
            std::cout << input[i] << ", ";
        }
        std::cout << " Output: " << output[0] << std::endl;
    }

    return 0;
}
