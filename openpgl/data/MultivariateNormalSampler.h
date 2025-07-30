//
// Created by fengshi on 7/31/25.
//

#ifndef MULTIVARIATENORMALSAMPLER_H
#define MULTIVARIATENORMALSAMPLER_H

#include <eigen/Eigen/Core>
#include <random>

struct MultivariateNormalSampler {
    MultivariateNormalSampler(int S, float *mean, float *cov, size_t seed);
    void draw(float *out);

    Eigen::Map<Eigen::VectorXf> mean;
    Eigen::MatrixXf transform;
    std::mt19937 gen;
};

#endif //MULTIVARIATENORMALSAMPLER_H
