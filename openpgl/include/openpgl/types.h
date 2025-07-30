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
    // PGL_SPATIAL_SPLIT_ROUNDROBIN,
    // PGL_SPATIAL_SPLIT_PPG,
    PGL_SPATIAL_SPLIT_VS,    // variance-scanning
    // PGL_SPATIAL_SPLIT_COVS,  // covariance-scanning
    PGL_SPATIAL_SPLIT_IGS,   // information-gain-scanning
    PGL_SPATIAL_SPLIT_FS,    // fluence-scanning
};

enum PGL_BASIS_FUNC_TYPE
{
    PGL_BASIS_FUNC_NN = 0,  // contribute directional samples to their nearest neighboring octahedral cell
    PGL_BASIS_FUNC_SPLAT,   // splat contribution of each samples to 9 neighboring cells
    PGL_BASIS_FUNC_DON_PCG,     // disjoint octave noise with PCG core
    PGL_BASIS_FUNC_DON_XI,      // disjoint octave noise with Xi-sequence core
    PGL_BASIS_FUNC_LATITUDE,
    PGL_BASIS_FUNC_LONGITUDE,
    PGL_BASIS_FUNC_CHECKERBOARD,  // similar to NN, using modulo instead of hash to map to the bin
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
    PGL_SPATIAL_CONFIDENCE_SIMULATION,     // estimate split confidence via MC simulation, assuming bin values follow multivariate normal distribution
};
