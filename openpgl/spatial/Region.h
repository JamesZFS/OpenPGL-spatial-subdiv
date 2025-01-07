// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleStatistics.h"
#include "../data/Signature.h"
#include "../data/CEStatistics.h"
// #include "../data/DivergenceStatistics.h"
#include "../openpgl_common.h"
#ifdef OPENPGL_RADIANCE_CACHES
#include "../directional/OutgoingRadianceHistogram.h"
#endif
#include "IRegion.h"

namespace openpgl
{
template <typename TDistribution, typename TTrainingStatistics>
struct Region : public IRegion {
    TDistribution distribution;
    BBox regionBounds;
    TTrainingStatistics trainingStatistics;
    SampleStatistics sampleStatistics;
    size_t numZeroValueSamples{0};
    bool splitFlag{false};

    struct SecondLevelCandidateSplit {
        float pos = std::numeric_limits<float>::quiet_NaN();
        Signature signaturesLR[2] {};  // for the lookahead children
        float energy = 0.0f;  // distance between signatures
        bool valid() const { return !std::isnan(pos); }

        void reset() {
            pos = std::numeric_limits<float>::quiet_NaN();
            signaturesLR[0].clear();
            signaturesLR[1].clear();
            energy = 0.0f;
        }

        void serialize(std::ostream &stream) const
        {
            stream.write(reinterpret_cast<const char *>(&pos), sizeof(pos));
            signaturesLR[0].serialize(stream);
            signaturesLR[1].serialize(stream);
            stream.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
        }

        void deserialize(std::istream &stream)
        {
            stream.read(reinterpret_cast<char *>(&pos), sizeof(pos));
            signaturesLR[0].deserialize(stream);
            signaturesLR[1].deserialize(stream);
            stream.read(reinterpret_cast<char *>(&energy), sizeof(energy));
        }
    };

    struct CandidateSplit {
        float pos = std::numeric_limits<float>::quiet_NaN();
        Signature signaturesLR[2] {};  // for the lookahead children
        float energy = 0.0f;  // distance between signatures
        SampleStatistics sampleStatisticsLR[2] {};  // to infer second-level candidate split

        SecondLevelCandidateSplit secondSplitsLR[2][3];

        bool valid() const { return !std::isnan(pos); }

        void reset() {
            pos = std::numeric_limits<float>::quiet_NaN();
            signaturesLR[0].clear();
            signaturesLR[1].clear();
            energy = 0.0f;

            sampleStatisticsLR[0].clear();
            sampleStatisticsLR[1].clear();
            for (uint8_t c = 0; c < 2; c++) {
                for (uint8_t i = 0; i < 3; i++) {
                    secondSplitsLR[c][i].reset();
                }
            }
        }

        void serialize(std::ostream &stream) const
        {
            stream.write(reinterpret_cast<const char *>(&pos), sizeof(pos));
            signaturesLR[0].serialize(stream);
            signaturesLR[1].serialize(stream);
            stream.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
            stream.write(reinterpret_cast<const char *>(&sampleStatisticsLR[0]), sizeof(SampleStatistics));
            stream.write(reinterpret_cast<const char *>(&sampleStatisticsLR[1]), sizeof(SampleStatistics));
            for (uint8_t c = 0; c < 2; c++) {
                for (uint8_t i = 0; i < 3; i++) {
                    secondSplitsLR[c][i].serialize(stream);
                }
            }
        }

        void deserialize(std::istream &stream)
        {
            stream.read(reinterpret_cast<char *>(&pos), sizeof(pos));
            signaturesLR[0].deserialize(stream);
            signaturesLR[1].deserialize(stream);
            stream.read(reinterpret_cast<char *>(&energy), sizeof(energy));
            stream.read(reinterpret_cast<char *>(&sampleStatisticsLR[0]), sizeof(SampleStatistics));
            stream.read(reinterpret_cast<char *>(&sampleStatisticsLR[1]), sizeof(SampleStatistics));
            for (uint8_t c = 0; c < 2; c++) {
                for (uint8_t i = 0; i < 3; i++) {
                    secondSplitsLR[c][i].deserialize(stream);
                }
            }
        }

        void copyFrom(const SecondLevelCandidateSplit &s) {
            pos = s.pos;
            signaturesLR[0] = s.signaturesLR[0];
            signaturesLR[1] = s.signaturesLR[1];
            energy = s.energy;
            sampleStatisticsLR[0].clear();
            sampleStatisticsLR[1].clear();
        }
    } candidateSplits[3];

    uint8_t bestSplitDim = 3;  // candidate split dim with the maximum energy
    CEStatistics ceStatistics;

    uint32_t depth = 0;  // depth in the tree
#ifdef OPENPGL_RADIANCE_CACHES
    OutgoingRadianceHistogram outRadianceHist;
#endif
    // bool valid{true};

    bool hasBestCandidateSplit() const {
        return bestSplitDim < 3 && candidateSplits[bestSplitDim].valid();
    }

    void clearCandidateSplits() {
        for (uint8_t i = 0; i < 3; i++) {
            candidateSplits[i].reset();
        }
        bestSplitDim = 3;
    }

    void inheritCandidateSplits(uint8_t splitDim, bool isRight) {
        for (uint8_t i = 0; i < 3; ++i) {
            candidateSplits[i].copyFrom(candidateSplits[splitDim].secondSplitsLR[isRight][i]);
        }
        bestSplitDim = 3;
    }

    void resetSignatures() {
        for (uint8_t i = 0; i < 3; i++) {
            candidateSplits[i].signaturesLR[0].clear();
            candidateSplits[i].signaturesLR[1].clear();
        }
    }

    const CandidateSplit &getBestCandidateSplit() const {
        OPENPGL_ASSERT(bestSplitDim < 3 && candidateSplits[bestSplitDim].valid());
        return candidateSplits[bestSplitDim];
    }

    inline const BBox &getRegionBounds() const
    {
        return regionBounds;
    }

    inline const BBox &getSampleBounds() const
    {
        return sampleStatistics.sampleBounds;
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
        sampleStatistics.serialize(stream);
#ifdef OPENPGL_RADIANCE_CACHES
        outRadianceHist.serialize(stream);
#endif
        stream.write(reinterpret_cast<const char *>(&numZeroValueSamples), sizeof(numZeroValueSamples));
        stream.write(reinterpret_cast<const char *>(&splitFlag), sizeof(splitFlag));
        candidateSplits[0].serialize(stream);
        candidateSplits[1].serialize(stream);
        candidateSplits[2].serialize(stream);
        stream.write(reinterpret_cast<const char *>(&bestSplitDim), sizeof(bestSplitDim));
        ceStatistics.serialize(stream);
        stream.write(reinterpret_cast<const char *>(&depth), sizeof(depth));
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&valid), sizeof(valid));
        distribution.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&regionBounds), sizeof(regionBounds));
        trainingStatistics.deserialize(stream);
        sampleStatistics.deserialize(stream);
#ifdef OPENPGL_RADIANCE_CACHES
        outRadianceHist.deserialize(stream);
#endif
        stream.read(reinterpret_cast<char *>(&numZeroValueSamples), sizeof(numZeroValueSamples));
        stream.read(reinterpret_cast<char *>(&splitFlag), sizeof(splitFlag));
        candidateSplits[0].deserialize(stream);
        candidateSplits[1].deserialize(stream);
        candidateSplits[2].deserialize(stream);
        stream.read(reinterpret_cast<char *>(&bestSplitDim), sizeof(bestSplitDim));
        ceStatistics.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&depth), sizeof(depth));
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
        ss << "\t sampleStatistics: " << sampleStatistics.toString() << std::endl;
        ss << "\t splitFlag: " << splitFlag << std::endl;
        ss << "\t valid: " << valid << std::endl;
        return ss.str();
    }

    bool operator==(const Region &b) const
    {
        bool equal = true;
        if (!sampleStatistics.operator==(b.sampleStatistics) || splitFlag != b.splitFlag)
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