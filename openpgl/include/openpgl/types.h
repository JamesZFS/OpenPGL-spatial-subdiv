#pragma once

enum PGL_SPATIAL_STRUCTURE_TYPE
{
    PGL_SPATIAL_STRUCTURE_KDTREE = 0
};

enum PGL_DIRECTIONAL_DISTRIBUTION_TYPE
{
    PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM = 0,
    PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE,
    PGL_DIRECTIONAL_DISTRIBUTION_VMM
};

enum PGL_SPATIAL_SPLIT_TYPE  // for PGL_SPATIAL_STRUCTURE_GENERIC_STREE
{
    PGL_SPATIAL_SPLIT_BASELINE = 0,
    PGL_SPATIAL_SPLIT_ROUNDROBIN,
    PGL_SPATIAL_SPLIT_PPG,
    PGL_SPATIAL_SPLIT_VS,    // variance-scanning
    PGL_SPATIAL_SPLIT_COVS,  // covariance-scanning
    PGL_SPATIAL_SPLIT_IGS,   // information-gain-scanning
    PGL_SPATIAL_SPLIT_FS,    // fluence-scanning
};

enum PGL_SPATIAL_CONTRIB_TYPE
{
    PGL_SPATIAL_CONTRIB_NN = 0,  // contribute directional samples to their nearest neighboring octahedral cell
    PGL_SPATIAL_CONTRIB_SPLAT,   // splat contribution of each samples to 9 neighboring cells
    PGL_SPATIAL_CONTRIB_BASIS,   // contribute directional samples to all signature bins using basis functions
    PGL_SPATIAL_CONTRIB_BASIS_XI,   // basis functions enhanced with xi-sequence
};

enum PGL_SPATIAL_DEFENSIVE_TYPE
{
    PGL_SPATIAL_DEFENSIVE_FIXED = 0,  // use fixed defensive sample count threshold
    PGL_SPATIAL_DEFENSIVE_SQRT,       // grow the defensive sample count with the speed of sqrt(iteration)
    PGL_SPATIAL_DEFENSIVE_PPG,        // set the defensive sample count to sqrt(iteration) only at iteration=2^k
};

enum PGL_SPATIAL_FILTER_TYPE
{
    PGL_SPATIAL_FILTER_NONE = 0,
    PGL_SPATIAL_FILTER_PERCENTAGE,
    PGL_SPATIAL_FILTER_DBOR,
    PGL_SPATIAL_FILTER_DBOR_ACCUM,
};

enum PGL_SPATIAL_CONFIDENCE_TYPE
{
    PGL_SPATIAL_CONFIDENCE_NONE = 0,  // always trust our metric
    PGL_SPATIAL_CONFIDENCE_RISK,      // trust if risk is low
    PGL_SPATIAL_CONFIDENCE_TTEST,
    PGL_SPATIAL_CONFIDENCE_TTEST_PER_BIN,  // only sum up the bins that pass the t-test
};
