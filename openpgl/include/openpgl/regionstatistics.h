#pragma once

#include "common.h"

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
    float diffCEStd = 0;
    bool hasCandidateSplit = false;
    pgl_point3f lowerBounds {0, 0, 0};
    pgl_point3f upperBounds {0, 0, 0};
};
