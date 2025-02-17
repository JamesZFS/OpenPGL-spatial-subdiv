#pragma once

#include "common.h"

extern uint32_t g_opgl_octahedral_resolution;
extern uint8_t g_opgl_signature_size;
extern uint8_t g_opgl_octave_min;
extern uint8_t g_opgl_octave_max;
extern float g_opgl_splat_sigma;
#define PGL_SIGNATURE_MAX_SIZE 8u

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
    float energy = 0;
    bool hasCandidateSplit = false;
    uint8_t splitDim = 3;
    float splitPos = 0;
    // pgl_point3f sampleMean {0, 0, 0};
    // pgl_vec3f sampleVariance {0, 0, 0};
    pgl_point3f lowerBounds {0, 0, 0};
    pgl_point3f upperBounds {0, 0, 0};
};

struct PGLDirectionalSignature
{
    float signature[PGL_SIGNATURE_MAX_SIZE] = {};
    float std[PGL_SIGNATURE_MAX_SIZE] = {};
    float numSamples = 0;
};

// From https://www.shadertoy.com/view/XlGcRh
inline std::pair<uint32_t, uint32_t> pgl_pcg2d(uint32_t x, uint32_t y) {
    x = x * 1664525u + 1013904223u;
    y = y * 1664525u + 1013904223u;

    x += y * 1664525u;
    y += x * 1664525u;

    x = x ^ (x>>16u);
    y = y ^ (y>>16u);

    x += y * 1664525u;
    y += x * 1664525u;

    x = x ^ (x>>16u);
    y = y ^ (y>>16u);

    return {x, y};
}

inline uint8_t pgl_get_signature_index(const pgl_direction &dir) {
    // 1. Convert the sample.direction into [0, 1] representation
    auto uv_ = pgl_vec2f(dir);  // [-1, 1]
    float x = uv_.x * 0.5f + 0.5f;  // [0, 1]
    float y = uv_.y * 0.5f + 0.5f;

    // 2. Find the histogram bin on the (conceptual) octahedral map
    uint32_t ix = std::clamp((uint32_t)(x * g_opgl_octahedral_resolution), 0u, g_opgl_octahedral_resolution - 1);
    uint32_t iy = std::clamp((uint32_t)(y * g_opgl_octahedral_resolution), 0u, g_opgl_octahedral_resolution - 1);

    // 3. Hash (ix, iy) to a single index between 0 and PGL_SIGNATURE_SIZE - 1
    // uint32_t hash = (2654435761 * ix) ^ (805459861 * iy);  // Instant-NGP
    uint32_t hash = pgl_pcg2d(ix, iy).first;
    return hash % g_opgl_signature_size;
}

inline uint8_t pgl_get_signature_index_jitter(const pgl_direction &dir, int dx, int dy) {
    // 1. Convert the sample.direction into [0, 1] representation
    auto uv_ = pgl_vec2f(dir);  // [-1, 1]
    float x = uv_.x * 0.5f + 0.5f;  // [0, 1]
    float y = uv_.y * 0.5f + 0.5f;

    // 2. Find the histogram bin on the (conceptual) octahedral map, jittered by (dx, dy)
    int ix = std::clamp((int)(x * g_opgl_octahedral_resolution) + dx, 0, (int)g_opgl_octahedral_resolution - 1);
    int iy = std::clamp((int)(y * g_opgl_octahedral_resolution) + dy, 0, (int)g_opgl_octahedral_resolution - 1);

    // 3. Hash (ix, iy) to a single index between 0 and PGL_SIGNATURE_SIZE - 1
    // uint32_t hash = (2654435761 * ix) ^ (805459861 * iy);  // Instant-NGP
    uint32_t hash = pgl_pcg2d(ix, iy).first;
    return hash % g_opgl_signature_size;
}
