// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl {

struct Embedding  // Directional embedding
{
    float bins[PGL_EMBEDDING_SIZE] = {};
    float normalizer = 0;

    void clear() {
        memset(bins, 0, sizeof(bins));
        normalizer = 0;
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        for (auto it = begin; it != end; ++it) {
            uint8_t idx = pgl_get_embedding_index(it->direction);
            bins[idx] += it->weight;
            ++normalizer;
        }
    }

    void addZeroSamples(size_t numSamples) {
        normalizer += (float) numSamples;
    }

    float getNumSamples() const {
        return normalizer;
    }

    float getEntry(uint8_t idx) const {
        return bins[idx] / normalizer;
    }

    explicit operator PGLDirectionalEmbedding() const {
        PGLDirectionalEmbedding embedding;
        for (uint8_t i = 0; i < PGL_EMBEDDING_SIZE; i++)
            embedding.embedding[i] = getEntry(i);
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
        stream.write(reinterpret_cast<const char *>(bins), sizeof(bins));
        stream.write(reinterpret_cast<const char *>(&normalizer), sizeof(normalizer));
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(bins), sizeof(bins));
        stream.read(reinterpret_cast<char *>(&normalizer), sizeof(normalizer));
    }
};

} // namespace openpgl
