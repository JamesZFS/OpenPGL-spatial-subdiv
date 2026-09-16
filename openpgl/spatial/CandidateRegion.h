// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// This file holds data structures required by the lookahead cells

#pragma once

#include "../data/SampleStatistics.h"
#include "../data/Signature.h"
#include "../openpgl_common.h"

namespace openpgl
{
struct CandidateSplit {
    float pivot {0};
    uint8_t dim : 2 {3};
    uint32_t lChildIdx : 30 {0};  // index into the candidate region storage
    float energy = 0.0f;  // distance between LR signatures
    bool updated = false;  // flag to indicate if this split has seen the latest samples

    bool valid() const { return dim < 3; }

    void reset() {
        pivot = 0;
        dim = 3;
        lChildIdx = 0;
        energy = 0.0f;
        updated = false;
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&pivot), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&pivot + 1), sizeof(uint32_t));  // dim and lChildIdx
        stream.write(reinterpret_cast<const char *>(&energy), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&updated), sizeof(bool));
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&pivot), sizeof(float));
        stream.read(reinterpret_cast<char *>(&pivot + 1), sizeof(uint32_t));  // dim and lChildIdx
        stream.read(reinterpret_cast<char *>(&energy), sizeof(float));
        stream.read(reinterpret_cast<char *>(&updated), sizeof(bool));
    }
};

struct CandidateRegion {
    Signature signature;
    SampleStatistics sampleStatistics;  // TODO: maybe merge signature and sampleStats into one
    CandidateSplit candidate;

    CandidateRegion &operator=(const CandidateRegion &) = delete;

    void serialize(std::ostream &stream) const
    {
        signature.serialize(stream);
        sampleStatistics.serialize(stream);
        candidate.serialize(stream);
    }

    void deserialize(std::istream &stream)
    {
        signature.deserialize(stream);
        sampleStatistics.deserialize(stream);
        candidate.deserialize(stream);
    }
};
}  // namespace openpgl