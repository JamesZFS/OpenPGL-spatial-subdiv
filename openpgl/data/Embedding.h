// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl {

struct Embedding  // Directional embedding
{
    float sum[PGL_EMBEDDING_SIZE] = {};  // sum of weights in that bin
    float m2[PGL_EMBEDDING_SIZE] = {};  // sum of squared weights in that bin
    float numSamples = 0;  // number of samples in all bins

    void clear() {
        memset(sum, 0, sizeof(sum));
        memset(m2, 0, sizeof(m2));
        numSamples = 0;
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        for (auto it = begin; it != end; ++it) {
            uint8_t idx = pgl_get_embedding_index(it->direction);
            sum[idx] += it->weight;
            m2[idx] += it->weight * it->weight;
            ++numSamples;
        }
    }

    void addZeroSamples(size_t numZeroSamples) {
        numSamples += (float) numZeroSamples;
    }

    float getNumSamples() const {
        return numSamples;
    }

    float getEntry(uint8_t idx) const {
        return sum[idx] / numSamples;
    }

    float getVariance(uint8_t idx) const {
        return m2[idx] / numSamples - sum[idx] * sum[idx] / (numSamples * numSamples);   // biased
    }

    explicit operator PGLDirectionalEmbedding() const {
        PGLDirectionalEmbedding embedding;
        for (uint8_t i = 0; i < PGL_EMBEDDING_SIZE; i++) {
            embedding.embedding[i] = getEntry(i);
            embedding.variance[i] = getVariance(i);
        }
        return embedding;
    }

    // L2 distance between two embeddings
    static float getDistanceL2(const Embedding &a, const Embedding &b) {
        float sum = 0;
        for (uint8_t i = 0; i < PGL_EMBEDDING_SIZE; i++) {
            float diff = a.getEntry(i) - b.getEntry(i);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // L1 distance between two embeddings
    static float getDistanceL1(const Embedding &a, const Embedding &b) {
        float sum = 0;
        for (uint8_t i = 0; i < PGL_EMBEDDING_SIZE; i++) {
            sum += std::abs(a.getEntry(i) - b.getEntry(i));
        }
        return sum;
    }

    // SMAPE: symmetric mean absolute percentage error
    // Has a range of [0, 2]
    static float getDistanceSMAPE(const Embedding &a, const Embedding &b) {
        float sum = 0;
        for (uint8_t i = 0; i < PGL_EMBEDDING_SIZE; i++) {
            float ai = a.getEntry(i), bi = b.getEntry(i);
            if (ai + bi > 0)
                sum += 2.0f * std::abs(ai - bi) / (ai + bi);
        }
        return sum / (float) PGL_EMBEDDING_SIZE;
    }

    static float getDistance(const Embedding &a, const Embedding &b) {
        return getDistanceSMAPE(a, b);
    }

    void serialize(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(sum), sizeof(sum));
        stream.write(reinterpret_cast<const char *>(m2), sizeof(m2));
        stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(numSamples));
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(sum), sizeof(sum));
        stream.read(reinterpret_cast<char *>(m2), sizeof(m2));
        stream.read(reinterpret_cast<char *>(&numSamples), sizeof(numSamples));
    }
};

} // namespace openpgl
