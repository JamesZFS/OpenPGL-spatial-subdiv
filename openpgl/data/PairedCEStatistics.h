// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#define COMPUTE_CE_STYLE 0  // 0: instantiating (parallaxed, cosined) guiding distributions at each sample location; 1: projecting samples to the center of the region

namespace openpgl {
struct PairedCEStatistics // Parent/child sufficient statistics for marginalized cross-entropies per region
{
    static float clampValue;

    float ap = 0;  // sum(- <phi> * log qp)
    float ac = 0;  // sum(- <phi> * log qc)
    float f = 0;  // sum( <phi> )
    float N = 0;  // number of samples
    float m2 = 0; // sum( (<phi> * (-log qp + log qc) )^2 )

    inline void clear() {
        ap = 0;
        ac = 0;
        f = 0;
        N = 0;
        m2 = 0;
    }

    /// Add a sample to the statistics
    /// @param weight: the Monte-Carlo weight, i.e., the Li estimate divided by sampling pdf
    /// @param qp: the *guiding* pdf of the parent
    /// @param qc: the *guiding* pdf of the child
    inline void addSample(float weight, float qp, float qc) {
        N++;
        weight = std::min(weight, clampValue);
        float parentSample = -weight * std::log(qp + 1e-3f);
        float childSample = -weight * std::log(qc + 1e-3f);
        ap += parentSample;
        ac += childSample;
        f += weight;
        float delta = parentSample - childSample;
        m2 += delta * delta;
    }

    inline void addZeroWeightSamples(size_t numSamples) {
        N += (float) numSamples;
    }

    inline float getFluence() const {
        return N > 0 ? f / N : 0;
    }

    inline float getParentCE() const {
        return f > 0 ? ap / f : 0;
    }

    inline float getChildCE() const {
        return f > 0 ? ac / f : 0;
    }

    inline float getReducedCE() const {
        return f > 0 ? (ap - ac) / f : 0;
    }

    // Assuming fluence is constant, estimating the std of reduced CE estimator
    inline float getStd() const {
        if (N <= 0) return 0;
        float a = ap - ac;
        return std::sqrt(m2 - a*a/N) / f;
    }

    inline float getNumSamples() const {
        return N;
    }

    static float weightedAverageReducedCE(const PairedCEStatistics &l, const PairedCEStatistics &r) {
        return (l.getReducedCE() * l.getNumSamples() + r.getReducedCE() * r.getNumSamples()) /
               (l.getNumSamples() + r.getNumSamples());
    }

    static float weightedAverageStd(const PairedCEStatistics &l, const PairedCEStatistics &r) {
        return (l.getStd() * l.getNumSamples() + r.getStd() * r.getNumSamples()) /
               (l.getNumSamples() + r.getNumSamples());
    }

    inline void decay(const float lambda) {
        ap *= lambda;
        ac *= lambda;
        f *= lambda;
        N *= lambda;
        m2 *= lambda;
    }

    inline void merge(const PairedCEStatistics &other) {
        ap += other.ap;
        ac += other.ac;
        f += other.f;
        N += other.N;
        m2 += other.m2;
    }

    inline PairedCEStatistics operator+(const PairedCEStatistics &other) const {
        PairedCEStatistics merged = *this;
        merged.merge(other);
        return merged;
    }

    inline bool isValid() const {
        bool valid = true;
        valid = valid && embree::isvalid(N) && N >= 0;

        valid = valid && embree::isvalid(ap);
        valid = valid && embree::isvalid(ac);
        valid = valid && embree::isvalid(f);
        valid = valid && embree::isvalid(m2);

        return valid;
    }

    std::string toString() const {
        std::stringstream ss;
        ss.precision(5);
        ss << "SampleStatistics:" << std::endl;
        ss << "N: " << N << std::endl;
        ss << "ap: " << ap << std::endl;
        ss << "ac: " << ac << std::endl;
        ss << "f: " << f << std::endl;
        ss << "m2: " << m2 << std::endl;
        ss << "fluence: " << getFluence() << std::endl;
        ss << "parent CE: " << getParentCE() << std::endl;
        ss << "child CE: " << getChildCE() << std::endl;
        ss << "Std: " << getStd() << std::endl;
        return ss.str();
    }

    void serialize(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&N), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&f), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&ap), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&ac), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&m2), sizeof(float));
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&N), sizeof(float));
        stream.read(reinterpret_cast<char *>(&f), sizeof(float));
        stream.read(reinterpret_cast<char *>(&ap), sizeof(float));
        stream.read(reinterpret_cast<char *>(&ac), sizeof(float));
        stream.read(reinterpret_cast<char *>(&m2), sizeof(float));
    }

    bool operator==(const PairedCEStatistics &b) const {
        bool equal = true;
        if (N != b.N || ap != b.ap || ac != b.ac || f != b.f || m2 != b.m2) {
            equal = false;
        }
        return equal;
    }
};

} // namespace openpgl
