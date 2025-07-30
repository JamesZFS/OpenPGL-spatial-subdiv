//
// Created by fengshi on 7/31/25.
//

#include "MultivariateNormalSampler.h"
#include <eigen/Eigen/Eigenvalues>

using namespace Eigen;

MultivariateNormalSampler::MultivariateNormalSampler(int S, float *mean, float *cov, size_t seed) : mean(mean, S), gen(seed) {
    Map<MatrixXf> covMat(cov, S, S);
    SelfAdjointEigenSolver<MatrixXf> eigenSolver(covMat);
    transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
}

void MultivariateNormalSampler::draw(float *out) {
    std::normal_distribution<float> dist;

    Eigen::VectorXf z(mean.size());
    for (int i = 0; i < z.size(); ++i)
        z(i) = dist(gen);

    Eigen::Map<Eigen::VectorXf> result(out, mean.size());
    result = mean + transform * z;
}
