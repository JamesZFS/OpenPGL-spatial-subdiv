// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#define COMPUTE_CE_STYLE 0  // 0: instantiating (parallaxed, cosined) guiding distributions at each sample location; 1: projecting samples to the center of the region

namespace openpgl {
struct CEStatistics // Sufficient statistics for marginalized cross-entropies per region
{
    static float clampValue;

    float a = 0;  // sum(- <phi> * log q)
    float f = 0;  // sum(<phi>)
    float N = 0;  // number of samples
    float m2 = 0; // sum( (<phi> * log q)^2 )

    inline void clear() {
        a = 0;
        f = 0;
        N = 0;
        m2 = 0;
    }

    /// Add a sample to the statistics
    /// @param weight: the Monte-Carlo weight, i.e., the Li estimate divided by sampling pdf
    /// @param pdf: the *guiding* pdf
    inline void addSample(float weight, float pdf) {
        N++;
        weight = std::min(weight, clampValue);
        float sample = -weight * std::log(pdf + 1e-3f);
        a += sample;
        f += weight;
        m2 += sample * sample;
        // a += std::min(-phi * std::log(pdf), clampValue);
        // f += std::min(phi, clampValue);
    }

    inline void addZeroWeightSamples(size_t numSamples) {
        N += (float) numSamples;
    }

    inline float getFluence() const {
        return N > 0 ? f / N : 0;
    }

    inline float getCE() const {
        return f > 0 ? a / f : 0;
    }

    // Assuming fluence is constant, estimating the std of CE estimator
    inline float getStd() const {
        if (N <= 0) return 0;
        // float varOneSample = m2 / N - a * a / (N * N);
        // float varNSample = varOneSample / N;
        // float fluence = getFluence();
        // return std::sqrt(varNSample) / fluence;
        return std::sqrt(m2 - a*a/N) / f;
    }

    inline float getNumSamples() const {
        return N;
    }

    static float weightedAverageCE(const CEStatistics &l, const CEStatistics &r) {
        return (l.getCE() * l.getNumSamples() + r.getCE() * r.getNumSamples()) /
               (l.getNumSamples() + r.getNumSamples());
    }

    static float weightedAverageFluence(const CEStatistics &l, const CEStatistics &r) {
        return (l.f + r.f) / (l.N + r.N);
    }

    inline void decay(const float lambda) {
        a *= lambda;
        f *= lambda;
        N *= lambda;
    }

    inline void merge(const CEStatistics &other) {
        a += other.a;
        f += other.f;
        N += other.N;
    }

    inline CEStatistics operator+(const CEStatistics &other) const {
        CEStatistics merged = *this;
        merged.merge(other);
        return merged;
    }

    inline bool isValid() const {
        bool valid = true;
        valid = valid && embree::isvalid(N) && N >= 0;

        valid = valid && embree::isvalid(a);
        valid = valid && embree::isvalid(f);

        return valid;
    }

    std::string toString() const {
        std::stringstream ss;
        ss.precision(5);
        ss << "SampleStatistics:" << std::endl;
        ss << "N: " << N << std::endl;
        ss << "a: " << a << std::endl;
        ss << "f: " << f << std::endl;
        ss << "fluence: " << getFluence() << std::endl;
        ss << "getCrossEntropy: " << getCE() << std::endl;
        return ss.str();
    }

    void serialize(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&N), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&f), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&a), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&m2), sizeof(float));
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&N), sizeof(float));
        stream.read(reinterpret_cast<char *>(&f), sizeof(float));
        stream.read(reinterpret_cast<char *>(&a), sizeof(float));
        stream.read(reinterpret_cast<char *>(&m2), sizeof(float));
    }

    bool operator==(const CEStatistics &b) const {
        bool equal = true;
        if (N != b.N || a != b.a || f != b.f || m2 != b.m2) {
            equal = false;
        }
        return equal;
    }
};

} // namespace openpgl
