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

// A ensemble of signatures (mixture of experts), where all signatures consume the same MC samples with a distinct configuration of bases.
// By this design, we are able to have different compressed representations of the (marginalized) radiance field at each region,
//  hopefully each of which capturing different features.
struct SignatureEnsemble {
    std::vector<Signature> data {1};  // at least one

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end, const std::vector<SignatureArguments> &ensembleConfig, bool multiplyCosine) {
        OPENPGL_ASSERT(ensembleConfig.size() == data.size());
        for (size_t i = 0; i < ensembleConfig.size(); ++i) {
            data[i].addSamples(begin, end, ensembleConfig[i], multiplyCosine);
        }
    }

    void addZeroSamples(size_t numZeroSamples) {
        for (auto &signature : data) {
            signature.addZeroSamples(numZeroSamples);
        }
    }

    void clear() {
        for (auto &signature : data) {
            signature.clear();
        }
    }

    void decay(float alpha) {
        for (auto &signature : data) {
            signature.decay(alpha);
        }
    }

    Signature &operator[](size_t i) {
        OPENPGL_ASSERT(i < data.size());
        return data[i];
    }

    const Signature &operator[](size_t i) const {
        OPENPGL_ASSERT(i < data.size());
        return data[i];
    }

    uint32_t getNumSignatures() const {
        return data.size();
    }

    void setNumSignatures(uint32_t numSignatures) {
        OPENPGL_ASSERT(numSignatures >= 1);
        data.resize(numSignatures);
    }

    float getNumSamples() const {
        return data[0].getNumSamples();
    }

    // The maximum distance across all signatures
    static float getDistance(const SignatureEnsemble &a, const SignatureEnsemble &b, float stdMultiplier) {
        float distance = 0.0f;
        for (size_t i = 0; i < a.data.size(); ++i) {
            distance = std::max(distance, Signature::getDistance(a.data[i], b.data[i], stdMultiplier));
        }
        return distance;
    }

    float getRisk() {
        float risk = 0.0f;
        for (const auto &signature : data) {
            risk = std::max(risk, signature.getRisk());
        }
        return risk;
    }

    void serialize(std::ostream &stream) const {
        uint32_t numSignatures = static_cast<uint32_t>(data.size());
        stream.write(reinterpret_cast<const char *>(&numSignatures), sizeof(numSignatures));
        for (const auto &signature : data) {
            signature.serialize(stream);
        }
    }

    void deserialize(std::istream &stream) {
        uint32_t numSignatures;
        stream.read(reinterpret_cast<char *>(&numSignatures), sizeof(numSignatures));
        data.resize(numSignatures);
        for (auto &signature : data) {
            signature.deserialize(stream);
        }
    }

};

struct SubdivisionData {
    SignatureEnsemble signatures;
    SampleStatistics sampleStatistics;

    float pivot {0};
    uint8_t dim : 2 {3};
    uint32_t lChildIdx : 30 {0};  // index into the candidate region storage

    float energy = 0.0f;  // distance between self and the parent node
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
        stream.read(reinterpret_cast<char *>(&risk), sizeof(float));
        stream.read(reinterpret_cast<char *>(&tValue), sizeof(float));
        stream.read(reinterpret_cast<char *>(&depth), sizeof(uint8_t));
        stream.read(reinterpret_cast<char *>(&updated), sizeof(bool));
    }
};

template <typename TDistribution, typename TTrainingStatistics>
struct Region : public IRegion {
    TDistribution distribution;
    BBox regionBounds;
    TTrainingStatistics trainingStatistics;
    size_t numZeroValueSamples{0};
    uint8_t splitFlag{0};  // a positive splitFlag indicates the number of splits to reach this region. This allows us to decay the directional model multiple times.

    SubdivisionData candidate;  // TODO: maybe use shared ptr?
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