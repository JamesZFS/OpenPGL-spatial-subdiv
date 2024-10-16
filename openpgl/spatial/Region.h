// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleStatistics.h"
#include "../data/CEStatistics.h"
#include "../openpgl_common.h"
#ifdef OPENPGL_RADIANCE_CACHES
#include "../directional/OutgoingRadianceHistogram.h"
#endif
#include "IRegion.h"

namespace openpgl
{
template <typename TDistribution, typename TTrainingStatistics>
struct Region : public IRegion
{
    TDistribution distribution;
    BBox regionBounds;
    TTrainingStatistics trainingStatistics;
    SampleStatistics sampleStatistics;
    size_t numZeroValueSamples{0};
    bool splitFlag{false};
    bool isLookahead{false};
    struct CandidateSplit {  // for non-lookahead regions
        float pos;
        uint8_t dim : 2;
        uint32_t dataIdx : 30;  // left child's index into the region storage, not the tree nodes!

        CandidateSplit() : pos(0), dim(3), dataIdx(0) {}

        bool valid() const { return dim < 3; }

        void serialize(std::ostream &stream) const
        {
            stream.write(reinterpret_cast<const char *>(&pos), sizeof(pos));
            auto dimAndNodeIdx = reinterpret_cast<const uint32_t *>(&pos + 1);
            stream.write(reinterpret_cast<const char *>(dimAndNodeIdx), sizeof(uint32_t));
        }

        void deserialize(std::istream &stream)
        {
            stream.read(reinterpret_cast<char *>(&pos), sizeof(pos));
            auto dimAndNodeIdx = reinterpret_cast<uint32_t *>(&pos + 1);
            stream.read(reinterpret_cast<char *>(dimAndNodeIdx), sizeof(uint32_t));
        }
    } candidateSplit;

    struct SelfAndParentCEStatistics {  // for lookahead regions
        CEStatistics self;
        CEStatistics parent;

        void serialize(std::ostream &stream) const
        {
            self.serialize(stream);
            parent.serialize(stream);
        }

        void deserialize(std::istream &stream)
        {
            self.deserialize(stream);
            parent.deserialize(stream);
        }
    } ceStatistics;
    uint32_t depth = 0;  // depth in the tree
#ifdef OPENPGL_RADIANCE_CACHES
    OutgoingRadianceHistogram outRadianceHist;
#endif
    // bool valid{true};
    void setLookahead() {
        isLookahead = true;
    }

    void unsetLookahead() {
        isLookahead = false;
        candidateSplit.dim = 3; // invalid
    }

    bool hasCandidateSplit() const {
        return !isLookahead && candidateSplit.valid();
    }

    void setCandidateSplit(uint8_t dim, float pos, uint32_t leftDataIdx) {
        isLookahead = false;
        candidateSplit.pos = pos;
        candidateSplit.dim = dim;
        candidateSplit.dataIdx = leftDataIdx;
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
        stream.write(reinterpret_cast<const char *>(&isLookahead), sizeof(isLookahead));
        ceStatistics.serialize(stream);
        candidateSplit.serialize(stream);
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
        stream.read(reinterpret_cast<char *>(&isLookahead), sizeof(isLookahead));
        ceStatistics.deserialize(stream);
        candidateSplit.deserialize(stream);
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