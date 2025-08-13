// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleStatistics.h"
#include "../data/Signature.h"
#include "../openpgl_common.h"
#ifdef OPENPGL_RADIANCE_CACHES
#include "../directional/OutgoingRadianceHistogram.h"
#endif
#include "IRegion.h"
// #include "CandidateRegion.h"

namespace openpgl
{

struct DirectionalStatistics {
    float sum[3] = {};
    float com[6] = {};
    float totalWeight = 0;
    float totalWeight2 = 0;

    static constexpr int idx_table[3][3] = {  // index into com
        {0, 1, 2},
        {1, 3, 4},
        {2, 4, 5}
    };

    void clear() {
        std::fill(sum, sum + 3, 0.0f);
        std::fill(com, com + 6, 0.0f);
        totalWeight = 0;
        totalWeight2 = 0;
    }

    void decay(float decayFactor) {
        for (int i = 0; i < 3; ++i) sum[i] *= decayFactor;
        for (int i = 0; i < 6; ++i) com[i] *= decayFactor;
        totalWeight *= decayFactor;
        totalWeight2 *= decayFactor;
    }

    DirectionalStatistics operator+(const DirectionalStatistics &other) const {
        DirectionalStatistics result;
        for (int i = 0; i < 3; ++i) result.sum[i] = sum[i] + other.sum[i];
        for (int i = 0; i < 6; ++i) result.com[i] = com[i] + other.com[i];
        result.totalWeight = totalWeight + other.totalWeight;
        result.totalWeight2 = totalWeight2 + other.totalWeight2;
        return result;
    }

    DirectionalStatistics operator-(const DirectionalStatistics &other) const {
        DirectionalStatistics result;
        for (int i = 0; i < 3; ++i) result.sum[i] = sum[i] - other.sum[i];
        for (int i = 0; i < 6; ++i) result.com[i] = com[i] - other.com[i];
        result.totalWeight = totalWeight - other.totalWeight;
        result.totalWeight2 = totalWeight2 - other.totalWeight2;
        return result;
    }

    // Accessing comomentum accumulator
    inline float &C(int i, int j) {
        return com[idx_table[i][j]];
    }

    inline float C(int i, int j) const {
        return com[idx_table[i][j]];
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        for (auto it = begin; it != end; ++it) {
            auto dir = pgl_vec3f(it->reprojectedDirection);
            float w = it->weight;
            sum[0] += w * dir[0], sum[1] += w * dir[1], sum[2] += w * dir[2];
            C(0, 0) += w * dir[0] * dir[0], C(0, 1) += w * dir[0] * dir[1], C(0, 2) += w * dir[0] * dir[2];
            C(1, 1) += w * dir[1] * dir[1], C(1, 2) += w * dir[1] * dir[2], C(2, 2) += w * dir[2] * dir[2];
            totalWeight += w;
            totalWeight2 += w * w;
        }
    }

    // Normalized mean direction
    pgl_vec3f getMean() const {
        float norm = std::sqrt(sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2]);
        return norm == 0 ? pgl_vec3f{0, 0, 0} : pgl_vec3f{sum[0] / norm, sum[1] / norm, sum[2] / norm};
    }

    pgl_vec3f get_d_bar() const {
        return totalWeight == 0 ? pgl_vec3f{0, 0, 0} : pgl_vec3f{sum[0] / totalWeight, sum[1] / totalWeight, sum[2] / totalWeight};
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

    // Standard error of the mean direction estimate
    float getStd() const {
        if (totalWeight == 0) return 0;
        auto d_bar = get_d_bar();  // mean direction, unnormalized
        float R2 = d_bar[0]*d_bar[0] + d_bar[1]*d_bar[1] + d_bar[2]*d_bar[2];
        auto mu = d_bar / std::sqrt(R2);
        float avg_muT_Com_mu =
            C(0, 0) / totalWeight * mu[0] * mu[0] + 2 * C(0, 1) / totalWeight * mu[0] * mu[1] + 2 * C(0, 2) / totalWeight * mu[0] * mu[2] +
            C(1, 1) / totalWeight * mu[1] * mu[1] + 2 * C(1, 2) / totalWeight * mu[1] * mu[2] + C(2, 2) / totalWeight * mu[2] * mu[2];
        float V = 1.0f - avg_muT_Com_mu;
        return std::sqrt((V * totalWeight2) / (totalWeight * totalWeight * R2));
    }

    // The half angle of the 100(1-alpha)% confidence interval cone centered at mean
    float getConfidenceInterval(float alpha) const {
        float sigma = getStd();
        return std::asin(std::sqrt(-std::log(alpha)) * sigma);
    }

    // Angle between the mean directions minus the two confidence cones' half angles
    static float getEffectiveAngle(const DirectionalStatistics &a, const DirectionalStatistics &b, float alpha) {
        auto muA = a.getMean(), muB = b.getMean();
        float dot = muA[0] * muB[0] + muA[1] * muB[1] + muA[2] * muB[2];
        return std::max(0.0f, std::acos(dot) - a.getConfidenceInterval(alpha) - b.getConfidenceInterval(alpha));
    }
};

// A ensemble of signatures (mixture of experts), where all signatures consume the same MC samples with a distinct configuration of bases.
// By this design, we are able to have different compressed representations of the (marginalized) radiance field at each region,
//  hopefully each of which capturing different features.
template<typename Alloc>
struct SignatureEnsemble {
    using SignatureType = Signature<Alloc>;
    SignatureType dataZero;  // use stack memory for non-MOE usages
    std::vector<SignatureType> dataRest;
    DirectionalStatistics dir; // mean direction statistics

    void init(int level, const std::vector<SignatureArguments> &ensembleConfig) {
        dataZero.init(level, ensembleConfig[0]);
        dataRest.resize(ensembleConfig.size() - 1);
        for (size_t i = 1; i < ensembleConfig.size(); ++i) {
            dataRest[i-1].init(level, ensembleConfig[i]);
        }
        dir.clear();
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end, const std::vector<SignatureArguments> &ensembleConfig, bool multiplyCosine) {
        OPENPGL_ASSERT(ensembleConfig.size() == getNumSignatures());
        dataZero.addSamples(begin, end, ensembleConfig[0], multiplyCosine);
        for (size_t i = 1; i < ensembleConfig.size(); ++i) {
            dataRest[i-1].addSamples(begin, end, ensembleConfig[i], multiplyCosine);
        }
        dir.addSamples(begin, end);
    }

    void addZeroSamples(size_t numZeroSamples) {
        dataZero.addZeroSamples(numZeroSamples);
        for (auto &signature : dataRest) {
            signature.addZeroSamples(numZeroSamples);
        }
    }

    void clear() {
        dataZero.clear();
        for (auto &signature : dataRest) {
            signature.clear();
        }
        dir.clear();
    }

    void decay(float alpha) {
        dataZero.decay(alpha);
        for (auto &signature : dataRest) {
            signature.decay(alpha);
        }
        dir.decay(alpha);
    }

    SignatureType &operator[](size_t i) {
        OPENPGL_ASSERT(i < getNumSignatures());
        return i == 0 ? dataZero : dataRest[i-1];
    }

    const SignatureType &operator[](size_t i) const {
        OPENPGL_ASSERT(i < getNumSignatures());
        return i == 0 ? dataZero : dataRest[i-1];
    }

    uint32_t getNumSignatures() const {
        return 1 + dataRest.size();
    }

    float getNumSamples() const {
        return dataZero.getNumSamples();
    }

    // The maximum distance across all signatures
    static float getDistance(const SignatureEnsemble &a, const SignatureEnsemble &b, float stdMultiplier) {
        float distance = SignatureType::getDistance(a.dataZero, b.dataZero, stdMultiplier);
        for (size_t i = 0; i < a.dataRest.size(); ++i) {
            distance = std::max(distance, SignatureType::getDistance(a.dataRest[i], b.dataRest[i], stdMultiplier));
        }
        return distance;
    }

    static float getSplitProbaMC(const SignatureEnsemble &a, const SignatureEnsemble &b, int numSimSamples, float T) {
        float stat = SignatureType::getSplitProbaMC(a.dataZero, b.dataZero, numSimSamples, T);
        for (size_t i = 0; i < a.dataRest.size(); ++i) {
            stat = std::max(stat, SignatureType::getSplitProbaMC(a.dataRest[i], b.dataRest[i], numSimSamples, T));
        }
        return stat;
    }

    float getRisk() {
        float risk = dataZero.getRisk();
        for (const auto &signature : dataRest) {
            risk = std::max(risk, signature.getRisk());
        }
        return risk;
    }

    void fillDirectionalStats(PGLDirectionalSignature &out) const {
        out.meanDir = dir.getMean();
        out.kappa = dir.getKappa();
        out.sigmaDir = dir.getStd();
    }

    void serialize(std::ostream &stream) const {
        uint32_t numSignatures = getNumSignatures();
        stream.write(reinterpret_cast<const char *>(&numSignatures), sizeof(numSignatures));
        dataZero.serialize(stream);
        for (const auto &signature : dataRest) {
            signature.serialize(stream);
        }
        stream.write(reinterpret_cast<const char *>(&dir), sizeof(dir));
    }

    void deserialize(std::istream &stream) {
        uint32_t numSignatures;
        stream.read(reinterpret_cast<char *>(&numSignatures), sizeof(numSignatures));
        dataZero.deserialize(stream);
        dataRest.resize(numSignatures - 1);
        for (auto &signature : dataRest) {
            signature.deserialize(stream);
        }
        stream.read(reinterpret_cast<char *>(&dir), sizeof(dir));
    }

};


template<typename Alloc>
struct SubdivisionData {
    SignatureEnsemble<Alloc> signatures;
    SampleStatistics sampleStatistics;

    float pivot {0};
    uint8_t dim : 2 {3};
    uint32_t lChildIdx : 30 {0};  // index into the candidate region storage

    float energy = 0.0f;  // distance between self and the parent node, or the statistics of the sufficient criterion
    float angularDistance = 0.0f;
    float risk = 0.0f;  // maximum value of std / mean per bin
    float tValue = 0.0f;
    uint8_t depth {0};
    bool updated = false;  // flag to indicate if this split has seen the latest samples

    bool hasSplit() const { return dim < 3; }

    void reset() {
        signatures.clear();
        sampleStatistics.clear();
        pivot = 0;
        dim = 3;
        lChildIdx = 0;
        energy = 0.0f;
        angularDistance = 0.0f;
        risk = 0.0f;
        tValue = 0.0f;
        depth = 0;
        updated = false;
    }

    void serialize(std::ostream &stream) const
    {
        signatures.serialize(stream);
        sampleStatistics.serialize(stream);
        stream.write(reinterpret_cast<const char *>(&pivot), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&pivot + 1), sizeof(uint32_t));  // dim and lChildIdx
        stream.write(reinterpret_cast<const char *>(&energy), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&angularDistance), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&risk), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&tValue), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&depth), sizeof(uint8_t));
        stream.write(reinterpret_cast<const char *>(&updated), sizeof(bool));
    }

    void deserialize(std::istream &stream)
    {
        signatures.deserialize(stream);
        sampleStatistics.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&pivot), sizeof(float));
        stream.read(reinterpret_cast<char *>(&pivot + 1), sizeof(uint32_t));  // dim and lChildIdx
        stream.read(reinterpret_cast<char *>(&energy), sizeof(float));
        stream.read(reinterpret_cast<char *>(&angularDistance), sizeof(float));
        stream.read(reinterpret_cast<char *>(&risk), sizeof(float));
        stream.read(reinterpret_cast<char *>(&tValue), sizeof(float));
        stream.read(reinterpret_cast<char *>(&depth), sizeof(uint8_t));
        stream.read(reinterpret_cast<char *>(&updated), sizeof(bool));
    }
};

template <typename TDistribution, typename TTrainingStatistics, typename Alloc>
struct Region : public IRegion {
    using SignatureAllocator = Alloc;
    TDistribution distribution;
    BBox regionBounds;
    TTrainingStatistics trainingStatistics;
    size_t numZeroValueSamples{0};
    uint8_t splitFlag{0};  // a positive splitFlag indicates the number of splits to reach this region. This allows us to decay the directional model multiple times.

    SubdivisionData<Alloc> candidate;  // TODO: maybe use shared ptr?
#ifdef OPENPGL_RADIANCE_CACHES
    OutgoingRadianceHistogram outRadianceHist;
#endif
    // bool valid{true};

    inline const BBox &getRegionBounds() const
    {
        return regionBounds;
    }

    inline const BBox &getSampleBounds() const
    {
        return candidate.sampleStatistics.sampleBounds;
    }

#ifdef OPENPGL_RADIANCE_CACHES
    Vector3 getOutgoingRadiance(const Vector3 dir) const override
    {
        return outRadianceHist.getOugoingRadiance(dir);
    }
#endif
    /*
    TDistribution getDistribution(Point3 samplePosition, const bool &useParallaxComp) const
    {
        TDistribution pDistribution = distribution;
        if(useParallaxComp)
        {
            Point3 pivotPosition = pDistribution._pivotPosition;
            pDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
        }
        return pDistribution;
    }
    */

    const TDistribution *getDistribution(Point3 samplePosition) const
    {
        return &distribution;
    }

    /*
    void getDistribution(TDistribution &pDistribution, Point3 samplePosition, const bool &useParallaxComp) const
    {
        pDistribution = distribution;
        if(useParallaxComp)
        {
            Point3 pivotPosition = pDistribution._pivotPosition;
            pDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
        }
        //return pDistribution;
    }
    */

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&valid), sizeof(valid));
        distribution.serialize(stream);
        stream.write(reinterpret_cast<const char *>(&regionBounds), sizeof(regionBounds));
        trainingStatistics.serialize(stream);
        candidate.sampleStatistics.serialize(stream);
#ifdef OPENPGL_RADIANCE_CACHES
        outRadianceHist.serialize(stream);
#endif
        stream.write(reinterpret_cast<const char *>(&numZeroValueSamples), sizeof(numZeroValueSamples));
        stream.write(reinterpret_cast<const char *>(&splitFlag), sizeof(splitFlag));
        candidate.serialize(stream);
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&valid), sizeof(valid));
        distribution.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&regionBounds), sizeof(regionBounds));
        trainingStatistics.deserialize(stream);
        candidate.sampleStatistics.deserialize(stream);
#ifdef OPENPGL_RADIANCE_CACHES
        outRadianceHist.deserialize(stream);
#endif
        stream.read(reinterpret_cast<char *>(&numZeroValueSamples), sizeof(numZeroValueSamples));
        stream.read(reinterpret_cast<char *>(&splitFlag), sizeof(splitFlag));
        candidate.deserialize(stream);
    }

    bool isValid() const
    {
        bool valid = true;
        valid = valid && distribution.isValid();
        valid = valid && trainingStatistics.isValid();
        //            valid = valid && sampleStatistics.isValid();
        return valid;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss.precision(5);
        ss << "Region:" << std::endl;
        ss << "\t regionBounds: " << regionBounds << std::endl;
        ss << "\t distribution: " << distribution.toString() << std::endl;
        ss << "\t trainingStatistics: " << trainingStatistics.toString() << std::endl;
        ss << "\t sampleStatistics: " << candidate.sampleStatistics.toString() << std::endl;
        ss << "\t splitFlag: " << splitFlag << std::endl;
        ss << "\t valid: " << valid << std::endl;
        return ss.str();
    }

    bool operator==(const Region &b) const
    {
        bool equal = true;
        if (!candidate.sampleStatistics.operator==(b.candidate.sampleStatistics) || splitFlag != b.splitFlag)
        {
            equal = false;
        }

        if (!distribution.operator==(b.distribution))
        {
            equal = false;
        }

        if (!trainingStatistics.operator==(b.trainingStatistics))
        {
            equal = false;
        }

        if (!(regionBounds == b.regionBounds))
        {
            equal = false;
        }

        return equal;
    }
};
}  // namespace openpgl