// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl {
struct CEStatistics // Sufficient statistics for marginalized cross-entropies per region
{
    float a = 0; // - <phi> * log q
    float f = 0; // <phi>
    float N = 0; // number of samples

    inline void clear() {
        a = 0;
        f = 0;
        N = 0;
    }

    /// Add a sample to the statistics
    /// @param phi: the Monte-Carlo weight, i.e., the Li estimate divided by sampling pdf
    /// @param pdf: the *guiding* pdf
    inline void addSample(float phi, float pdf) {
        N++;
        a += -phi * std::log(pdf);
        f += phi;
    }

    inline void addZeroWeightSamples(size_t numSamples) {
        N += (float) numSamples;
    }

    inline float getFluence() const {
        return N > 0 ? f / N : 0;
    }

    inline float getCE() const {
        return a / (f + 1e-3f);
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
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&N), sizeof(float));
        stream.read(reinterpret_cast<char *>(&f), sizeof(float));
        stream.read(reinterpret_cast<char *>(&a), sizeof(float));
    }

    bool operator==(const CEStatistics &b) const {
        bool equal = true;
        if (N != b.N || a != b.a || f != b.f) {
            equal = false;
        }
        return equal;
    }
};

} // namespace openpgl
