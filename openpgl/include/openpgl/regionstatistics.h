#pragma once

#include "common.h"
#include "defines.h"

/// Helper struct to communicate between OpenPGL and the renderer/visualizer to output debug information.
struct PGLRegionStatistics
{
    uint32_t id = -1;
    bool removed = false;
    uint32_t numSamples = 0;
    uint32_t numZeroValueSamples = 0;
    uint32_t depth = 0;
    float fluence = 0;
    float energy = 0;
    float angularDistance = 0;
    float risk = 0;
    float tValue = 0;
    bool hasCandidateSplit = false;
    uint8_t splitDim = 3;
    float splitPos = 0;
    // pgl_point3f sampleMean {0, 0, 0};
    pgl_vec3f sampleVariance {0, 0, 0};
    pgl_point3f lowerBounds {0, 0, 0};
    pgl_point3f upperBounds {0, 0, 0};
    float dborMean = 0;
    float dborStd = 0;
};

struct PGLDirectionalSignature
{
    float signature[PGL_SIGNATURE_MAX_SIZE] = {};
    float std[PGL_SIGNATURE_MAX_SIZE] = {};
    uint8_t S = 0;
    float numSamples = 0;
    // Mean direction estimate
    pgl_vec3f meanDir{0, 0, 0};
    float kappa = 0;  // of the VMF
    float sigmaDir = 0;
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
