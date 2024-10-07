#pragma once

#include "common.h"

/// Helper struct to communicate between OpenPGL and the renderer/visualizer to output debug information.
struct PGLRegionStatistics
{
    uint32_t id = -1;
    // TODO: need flag to tell if it's a leaf, lookahead, or inactive region
    uint32_t numSamples = 0;
    uint32_t numZeroValueSamples = 0;
    uint32_t depth = 0;
    float fluence = 0;
    float crossEntropy = 0;
};
