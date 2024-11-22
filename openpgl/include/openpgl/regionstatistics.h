#pragma once

#include "common.h"

#define PGL_OCTAHEDRAL_MAP_RESOLUTION 32u
#define PGL_EMBEDDING_SIZE 8u

/// Helper struct to communicate between OpenPGL and the renderer/visualizer to output debug information.
struct PGLRegionStatistics
{
    uint32_t id = -1;
    bool removed = false;
    uint32_t numSamples = 0;
    uint32_t numZeroValueSamples = 0;
    uint32_t depth = 0;
    float fluence = 0;
    float crossEntropy = 0;
    bool hasCandidateSplit = false;
    pgl_point3f lowerBounds {0, 0, 0};
    pgl_point3f upperBounds {0, 0, 0};
};

struct PGLDirectionalEmbedding
{
    float embedding[PGL_EMBEDDING_SIZE] = {};
};

inline uint8_t pgl_get_embedding_index(const pgl_direction &dir) {
    // 1. Convert the sample.direction into [0, 1] representation
    auto uv_ = pgl_vec2f(dir);  // [-1, 1]
    float x = uv_.x * 0.5f + 0.5f;  // [0, 1]
    float y = uv_.y * 0.5f + 0.5f;

    // 2. Find the histogram bin on the (conceptual) octahedral map
    uint32_t ix = std::clamp((uint32_t)(x * PGL_OCTAHEDRAL_MAP_RESOLUTION), 0u, PGL_OCTAHEDRAL_MAP_RESOLUTION - 1);
    uint32_t iy = std::clamp((uint32_t)(y * PGL_OCTAHEDRAL_MAP_RESOLUTION), 0u, PGL_OCTAHEDRAL_MAP_RESOLUTION - 1);

    // 3. Hash (ix, iy) to a single index between 0 and EMBEDDING_SIZE - 1
    uint32_t hash = (2654435761 * ix) ^ (805459861 * iy);
    return hash % PGL_EMBEDDING_SIZE;
}
