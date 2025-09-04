// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleStatistics.h"
#include "../data/Signature.h"
#include "../openpgl_common.h"
#include <tbb/concurrent_vector.h>
#ifdef OPENPGL_RADIANCE_CACHES
#include "../directional/OutgoingRadianceHistogram.h"
#endif
#include "IRegion.h"

namespace openpgl
{

struct SubdivisionData {
    // Necessary memory: 10 - 20 floats ~ 40 - 80 Bytes
    Signature signature;  // 3 - 6 floats
    union {  // can only have sampleStatistics before a split
        SampleStatistics sampleStatistics;  // essentially 7 floats
        struct {
            float pivot {0};
            uint8_t dim : 2 {0};
            uint32_t lChildIdx : 30 {0};  // index into the candidate region storage
        };
    };
    static_assert(sizeof(SampleStatistics) >= 3 * 4);

    bool updated = false;  // flag to indicate if this split has seen the latest samples

    SubdivisionData() : signature(), sampleStatistics(), updated(false) {}

    SubdivisionData(const SubdivisionData &other) {
        signature = other.signature;
        sampleStatistics = other.sampleStatistics;
        updated = other.updated;
    }

    SubdivisionData &operator=(const SubdivisionData &other) {
        signature = other.signature;
        sampleStatistics = other.sampleStatistics;
        updated = other.updated;
        return *this;
    }


    bool hasSplit() const { return sampleStatistics.numSamples < 0; }  // magic

    void setSplit(uint8_t dim, float pivot) {
        this->dim = dim;
        this->pivot = pivot;
        sampleStatistics.numSamples = -1;  // flag for having split
    }

    void reset() {
        signature.clear();
        sampleStatistics.clear();
        updated = false;
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&signature), sizeof(signature));
        sampleStatistics.serialize(stream);
        stream.write(reinterpret_cast<const char *>(&updated), sizeof(bool));
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&signature), sizeof(signature));
        sampleStatistics.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&updated), sizeof(bool));
    }
};

template <typename TDistribution, typename TTrainingStatistics>
struct Region : public IRegion {
    TDistribution distribution;
    BBox regionBounds;
    TTrainingStatistics trainingStatistics;
    Vector3 regionPivot;
    size_t numZeroValueSamples{0};
    uint8_t splitFlag{0};  // a positive splitFlag indicates the number of splits to reach this region. This allows us to decay the directional model multiple times.

    SubdivisionData candidate;
#ifdef OPENPGL_RADIANCE_CACHES
    OutgoingRadianceHistogram outRadianceHist;
#endif
    // bool valid{true};

    inline const BBox &getRegionBounds() const
    {
        return regionBounds;
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

    size_t getHeapMemory() const {
        return distribution.getHeapMemory() + trainingStatistics.getHeapMemory();
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&valid), sizeof(valid));
        stream.write(reinterpret_cast<const char *>(&initialized), sizeof(initialized));
        distribution.serialize(stream);
        stream.write(reinterpret_cast<const char *>(&regionBounds), sizeof(regionBounds));
        stream.write(reinterpret_cast<const char *>(&regionPivot), sizeof(regionPivot));
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
        stream.read(reinterpret_cast<char *>(&initialized), sizeof(initialized));
        distribution.deserialize(stream);
        stream.read(reinterpret_cast<char *>(&regionBounds), sizeof(regionBounds));
        stream.read(reinterpret_cast<char *>(&regionPivot), sizeof(regionPivot));
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
        if (initialized)
        {
            valid = valid && distribution.isValid();
            valid = valid && trainingStatistics.isValid();
            //            valid = valid && sampleStatistics.isValid();
        }
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

class CandidateRegionStorage {
public:
    using iterator = tbb::concurrent_vector<SubdivisionData>::iterator;

    void reserve(size_t n) {
        m_data.reserve(n);
    }

    void clear() {
        m_data.clear();
    }
    
    size_t size() const { return m_data.size(); }

    iterator begin() { return m_data.begin(); }

    iterator end() { return m_data.end(); }

    SubdivisionData &operator[](size_t idx) {
        return m_data[idx];
    }

    const SubdivisionData &operator[](size_t idx) const {
        return m_data[idx];
    }

    size_t add_pair() {
        size_t idx = m_freeIndexStack.pop();  // Check if there is a recyclable pair before allocating new
        if (idx != -1) return idx;
        return std::distance(m_data.begin(), m_data.grow_by(2));
    }

    size_t append() {
        return std::distance(m_data.begin(), m_data.emplace_back());
    }

    void recycle_pair(size_t idx) {
        m_data[idx].reset();
        m_data[idx+1].reset();
        m_freeIndexStack.push(idx);
    }

private:
    tbb::concurrent_vector<SubdivisionData> m_data;

    struct ConcurrentIndexStack {
        std::vector<size_t> indices;
        std::mutex mutex;

        void push(size_t i) {
            std::lock_guard lock(mutex);
            indices.push_back(i);
        }

        size_t pop() {
            std::lock_guard lock(mutex);
            if (indices.empty()) return -1;
            size_t ret = indices.back();
            indices.pop_back();
            return ret;
        }
    } m_freeIndexStack;
};

}  // namespace openpgl