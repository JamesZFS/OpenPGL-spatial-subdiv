// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace openpgl {

// CDF of standard normal distribution
static inline float Phi(float x) {
    return 0.5f * (std::erf(x / std::sqrt(2)) + 1);
}

// y in (0, 1)
static inline float PhiInv(float y) {
    float lo = -1e5f, hi = +1e5f;
    while (hi - lo > 1e-3f) {
        float mi = (lo + hi) / 2.f;
        if (Phi(mi) <= y) lo = mi;
        else hi = mi;
    }
    return lo;
}

struct Signature  // Directional signature
{
    float totalWeight = 0;
    float totalWeight2 = 0;
    float numSamples = 0;

    // Directional statistics
    float dirSum[3] = {};

    void clear() {
        std::fill_n(dirSum, 3, 0.0f);
        totalWeight = 0;
        totalWeight2 = 0;
        numSamples = 0;
    }

    void decay(float decayFactor) {
        for (int i = 0; i < 3; ++i) dirSum[i] *= decayFactor;
        totalWeight *= decayFactor;
        totalWeight2 *= decayFactor;
        numSamples *= decayFactor;
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        for (auto it = begin; it != end; ++it) {
            auto dir = pgl_vec3f(it->reprojectedDirection);
            float w = it->weight;
            dirSum[0] += w * dir[0], dirSum[1] += w * dir[1], dirSum[2] += w * dir[2];
            totalWeight += w;
            totalWeight2 += w * w;
        }
        numSamples += std::distance(begin, end);
    }

    void addZeroSamples(size_t numZeroSamples) {
        numSamples += (float) numZeroSamples;
    }

    float getFluence() const {
        return totalWeight / numSamples;
    }

    float getFluenceVar() const {
        float mu = totalWeight / numSamples;
        float var = totalWeight2 / numSamples - mu * mu;
        return var / numSamples;
    }

    float getFluenceStd() const {
        return std::sqrt(getFluenceVar());
    }

    explicit operator PGLDirectionalSignature() const {
        PGLDirectionalSignature out;
        out.signature = getFluence();
        out.std = getFluenceStd();
        out.numSamples = numSamples;
        out.meanDir = getMeanDir();
        out.kappa = getKappa();
        out.sigmaDir = getMeanDirStd();
        out.kappa_eff = getKappaEff();
        return out;
    }

    // [0, +inf]
    static float getFluenceSplitConfidence(const Signature &A, const Signature &B, float T) {
        // D is A - B
        float D_numSamples = A.numSamples - B.numSamples;
        float D_mean = (A.totalWeight - B.totalWeight) / D_numSamples;
        float D_s2 = ((A.totalWeight2 - B.totalWeight2) / D_numSamples - D_mean * D_mean) / D_numSamples;
        float B_mean = B.getFluence();
        float B_s2 = B.getFluenceVar();

        float wB_n = +(1-T) * B.numSamples - A.numSamples, wD_n = +(1-T) * D_numSamples;
        float wB_p = -(1+T) * B.numSamples + A.numSamples, wD_p = -(1+T) * D_numSamples;

        float criterion_n = (wB_n * B_mean + wD_n * D_mean) / std::sqrt(wB_n*wB_n * B_s2 + wD_n*wD_n * D_s2);
        float criterion_p = (wB_p * B_mean + wD_p * D_mean) / std::sqrt(wB_p*wB_p * B_s2 + wD_p*wD_p * D_s2);
        return std::max(criterion_n, criterion_p);
    }

    // Normalized mean direction
    pgl_vec3f getMeanDir() const {
        float norm = std::sqrt(dirSum[0]*dirSum[0] + dirSum[1]*dirSum[1] + dirSum[2]*dirSum[2]);
        return norm == 0 ? pgl_vec3f{0, 0, 0} : pgl_vec3f{dirSum[0] / norm, dirSum[1] / norm, dirSum[2] / norm};
    }

    pgl_vec3f get_d_bar() const {
        return totalWeight == 0 ? pgl_vec3f{0, 0, 0} : pgl_vec3f{dirSum[0] / totalWeight, dirSum[1] / totalWeight, dirSum[2] / totalWeight};
    }

    float get_R_bar() const {
        return std::sqrt(get_R_bar2());
    }

    float get_R_bar2() const {
        auto d_bar = get_d_bar();
        return d_bar[0]*d_bar[0] + d_bar[1]*d_bar[1] + d_bar[2]*d_bar[2];
    }

    float getKappa() const {
        float R2 = get_R_bar2();
        R2 = std::min(R2, 1.0f - 1e-6f); // prevent div by 0
        float R = std::sqrt(R2);
        return R * (3 - R2) / (1 - R2);  // approximation
    }

    float getKappaEff() const {  // approximate 1 / sigma^2 without using Comoment
        float R2 = get_R_bar2();
        R2 = std::min(R2, 1.0f - 1e-6f); // prevent div by 0
        float KappaEff = 0.5f * R2 * (3 - R2) / (1 - R2);
        return KappaEff * (totalWeight * totalWeight) / totalWeight2;
    }

    // Standard error of the mean direction estimate
    float getMeanDirStd() const {
        float kappaEff = getKappaEff();
        return std::sqrt(1.f / kappaEff);
    }

    // The half angle of the 100(1-alpha)% confidence interval cone centered at mean
    float getConfidenceInterval(float alpha) const {
        float sigma = getMeanDirStd();
        return std::asin(std::sqrt(-std::log(alpha)) * sigma);
    }

    // Angle between the mean directions minus the two confidence cones' half angles
    static float getAngularDistance(const Signature &A, const Signature &B, float alpha) {
        auto muA = A.getMeanDir(), muB = B.getMeanDir();
        float dot = muA[0] * muB[0] + muA[1] * muB[1] + muA[2] * muB[2];
        return std::max(0.0f, std::acos(dot) - A.getConfidenceInterval(alpha) - B.getConfidenceInterval(alpha));
    }

    // Approximate the split probability P(w_a dot w_b <= cos deltaTheta) using series sum
    static float getAngularSplitConfidence(const Signature &A, const Signature &B, float deltaTheta) {
        float x = std::cos(deltaTheta);
        auto muA = A.getMeanDir(), muB = B.getMeanDir();
        float z = muA[0] * muB[0] + muA[1] * muB[1] + muA[2] * muB[2];
        z = std::clamp(z, -1.f, 1.f);
        float kappaA, kappaB;
        kappaA = A.getKappaEff();
        kappaB = B.getKappaEff();
        float kappa = kappaA * kappaB / (kappaA + kappaB);  // effective kappa

        // Compute the series
        float s = 0.5f * (x+1);  // zero-order term
        constexpr int lMax = 50;
        for (int l = 1; l <= lMax; ++l) {
            s += 0.5f * std::exp(-float(l*(l+1)) / (2.f*kappa)) * std::legendre(l, z) * (std::legendre(l+1, x) - std::legendre(l-1, x));
        }
        return std::clamp(s, 0.f, 1.f);
    }
};

} // namespace openpgl
