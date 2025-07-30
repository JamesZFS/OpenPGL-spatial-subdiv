// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>
#else
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#endif
#include "MultivariateNormalSampler.h"
#include "../include/openpgl/signaturearguments.h"

namespace openpgl {

template<typename Alloc>
struct Signature  // Directional signature
{
    std::vector<float, Alloc> sum;  // sum of weights in that bin
    std::vector<float, Alloc> com;  // comomentum accumulator, storing only the upper triangle
    float numSamples = 0;  // number of samples in all bins, or sum of sample weights

    void init(int level, const SignatureArguments &config) {
        int size = 0;
        if (level <= PGL_SIGNATURE_FULL_RES_LEVEL) {  // First k levels: full resolution
            size = config.numBins;
        } else {  // Otherwise: one bin
            size = 1;
        }
        // No need to reset the signature if change from one bin to one bin again
        // if (sum.size() == size) return;
        sum.clear();
        com.clear();
        sum.resize(size, 0);
        com.resize((size * (size + 1)) >> 1, 0);
        numSamples = 0;
    }

    void clear() {
        std::fill(sum.begin(), sum.end(), 0.0f);
        std::fill(com.begin(), com.end(), 0.0f);
        numSamples = 0;
    }

    // Merge
    Signature operator+(const Signature &other) const {
        Signature result;
        result.sum.resize(sum.size());
        result.com.resize(com.size());
        result.numSamples = numSamples + other.numSamples;

        for (size_t i = 0; i < sum.size(); ++i) {
            result.sum[i] = sum[i] + other.sum[i];
        }

        for (size_t i = 0; i < com.size(); ++i) {
            result.com[i] = com[i] + other.com[i];
        }

        return result;
    }

    // Subtract
    Signature operator-(const Signature &other) const {
        Signature result;
        result.sum.resize(sum.size());
        result.com.resize(com.size());
        result.numSamples = numSamples - other.numSamples;

        for (size_t i = 0; i < sum.size(); ++i) {
            result.sum[i] = sum[i] - other.sum[i];
        }

        for (size_t i = 0; i < com.size(); ++i) {
            result.com[i] = com[i] - other.com[i];
        }

        return result;
    }

    // Accessing comomentum accumulator
    inline float &C(int i, int j) {
        if (j < i) std::swap(i, j);
        int index = (((2 * sum.size() - i + 1) * i) >> 1) + j - i;
        OPENPGL_ASSERT(0 <= index && index < com.size());
        return com[index];
    }

    inline float C(int i, int j) const {
        if (j < i) std::swap(i, j);
        int index = (((2 * sum.size() - i + 1) * i) >> 1) + j - i;
        OPENPGL_ASSERT(0 <= index && index < com.size());
        return com[index];
    }

    // Convert the direction into [0, 1] representation on the octahedral map
    inline static pgl_vec2f dir_to_oct(const pgl_direction &dir) {
        auto uv = pgl_vec2f(dir);  // [-1, 1]
        return {uv.x * 0.5f + 0.5f, uv.y * 0.5f + 0.5f};
    }

    inline static pgl_vec2f dir_to_spherical(const pgl_direction &dir) {
        auto cartesian = pgl_vec3f(dir);
        float theta = std::acos(cartesian.z);
        float phi = std::atan2(cartesian.y, cartesian.x);
        if (phi < 0) phi += 2 * M_PI;
        return {theta / M_PI, phi / (2 * M_PI)};  // TODO: optimize by returning the cosines and sines
    }

    //A pseudorandom number generator with a seed consisting of 3 uints
    inline static uint32_t pcg_3d(uint32_t x, uint32_t y, uint32_t z) {
        x ^= 12312u;
        // Taken from: https://www.shadertoy.com/view/XlGcRh
        x = x * 1664525u + 1013904223u;
        y = y * 1664525u + 1013904223u;
        z = z * 1664525u + 1013904223u;
        x += y * z;
        y += z * x;
        z += x * y;
        x ^= x >> 16u;
        y ^= y >> 16u;
        z ^= z >> 16u;
        x += y * z;
        y += z * x;
        z += x * y;
        return x;
    }

    // Inserts one 0-bit between any two of the 16 low bits of x
    inline static uint32_t part_1_by_1(uint32_t x) {
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x &= 0xffffu;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x << 8u)) & 0xff00ffu;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x << 4u)) & 0xf0f0f0fu;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x << 2u)) & 0x33333333u;
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x = (x ^ (x << 1u)) & 0x55555555u;
        return x;
    }


    // Creates a Morton code from two 8-bit integers (costs more than morton_8())
    inline static uint32_t morton_16(uint32_t x, uint32_t y) {
        return (part_1_by_1(y) << 1u) ^ part_1_by_1(x);
    }


    // Inverse of xi() for max_depth = 16 (or maybe not)
    inline static uint32_t invert_xi_16(uint32_t x, uint32_t y) {
        x ^= 2631929843u, y ^= 3492732422u;
        uint32_t z = morton_16(x >> 16u, y >> 16u);
        constexpr uint32_t U[4] = {0u, 1790330939u, 2934368918u, 3293618861u};
        uint32_t seq_no = 0u;
        for (uint32_t bit = 0u; bit < 32u; bit += 2u)
            seq_no ^= U[(z >> (30u - bit)) & 3u] << bit;
        return seq_no;
    }


    // Inverse of xi() (or maybe not)
    inline static uint32_t invert_xi(uint32_t x, uint32_t y, uint32_t max_depth) {
        uint32_t mask = (1u << (2u * max_depth)) - 1u;
        mask = (max_depth == 16u) ? 0xffffffffu : mask;
        return invert_xi_16(x << (32u - max_depth), y << (32u - max_depth)) & mask;
    }

    // Maps a texel index and an octave index to a bin index. Only the octave least
    // significant bits of the texel index should be non-zero. octave must be 8 or
    // less.
    uint32_t static get_bin(uint32_t x, uint32_t y, uint32_t octave, uint32_t log2_bin_count) {
        uint32_t seq_no = invert_xi(x, y, 2u * octave);
        uint32_t bin = (seq_no >> (4u * octave - log2_bin_count)) & ((1u << log2_bin_count) - 1u);
        return bin;
    }

    inline static float mix(float a, float b, float t) {
        return (1 - t) * a + t * b;
    }

    inline static float fract(float x) {
        return x - std::floor(x);
    }

    // Implements wrapping of k-bit indices in a way that is compatible with
    // octahedral maps
    inline static void wrap(uint32_t &x, uint32_t &y, uint32_t res) {
        const uint32_t hres = res >> 1;
        if (x > hres && (y == 0u || y == res)) {
            x = res - x;
        }
        if (y > hres && (x == 0u || x == res)) {
            y = res - y;
        }
        x &= res - 1u, y &= res - 1u;
    }

    // Wrapping for splatting
    inline static void wrap_splat(int &x, int &y, uint32_t res) {
        if (x < 0 || x >= res) {
            x = std::clamp(x, 0, (int)res - 1);
            y = res - 1 - y;
        }
        if (y < 0 || y >= res) {
            y = std::clamp(y, 0, (int)res - 1);
            x = res - 1 - x;
        }
    }

    inline uint8_t get_signature_index_nn(const pgl_direction &dir, uint32_t res, uint8_t S) {
        // 1. Convert the sample.direction into [0, 1] representation
        auto uv_ = pgl_vec2f(dir);  // [-1, 1]
        float x = uv_.x * 0.5f + 0.5f;  // [0, 1]
        float y = uv_.y * 0.5f + 0.5f;

        // 2. Find the histogram bin on the (conceptual) octahedral map
        uint32_t ix = std::clamp((uint32_t)(x * res), 0u, res - 1);
        uint32_t iy = std::clamp((uint32_t)(y * res), 0u, res - 1);

        // 3. Hash (ix, iy) to a single index between 0 and PGL_SIGNATURE_SIZE - 1
        // uint32_t hash = (2654435761 * ix) ^ (805459861 * iy);  // Instant-NGP
        uint32_t hash = pgl_pcg2d(ix, iy).first;
        return hash % S;
    }

    inline uint8_t get_signature_index_checkerboard(const pgl_direction &dir, uint32_t res, uint8_t S) {
        // 1. Convert the sample.direction into [0, 1] representation
        auto uv_ = pgl_vec2f(dir);  // [-1, 1]
        float x = uv_.x * 0.5f + 0.5f;  // [0, 1]
        float y = uv_.y * 0.5f + 0.5f;

        // 2. Find the histogram bin on the (conceptual) octahedral map
        uint32_t ix = std::clamp((uint32_t)(x * res), 0u, res - 1);
        uint32_t iy = std::clamp((uint32_t)(y * res), 0u, res - 1);

        // 3. Hash (ix, iy) to a single index between 0 and PGL_SIGNATURE_SIZE - 1
        uint32_t index = ix + iy * res;
        return index % S;
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end, const SignatureArguments &config, bool multiplyCosine) {
        numSamples += std::distance(begin, end);
        if (sum.size() == 1) {  // One bin
            for (auto it = begin; it != end; ++it) {
                float w = it->weight;
                if (multiplyCosine) w *= it->cosineTerm;
                sum[0] += w;
                com[0] += w * w;
            }
            return;
        }
        const uint8_t S = config.numBins;
        switch (config.basisType) {
            case PGL_BASIS_FUNC_NN: {
                const uint32_t res = config.getResolution();
                for (auto it = begin; it != end; ++it) {
                    uint8_t idx = get_signature_index_nn(it->reprojectedDirection, res, S);
                    float w = it->weight;
                    if (multiplyCosine) {
                        w *= it->cosineTerm;
                        // pgl_vec3f dir = it->reprojectedDirection;
                        // pgl_vec3f normal = it->normal;
                        // w *= std::max(0.0f, dir.x * normal.x + dir.y * normal.y + dir.z * normal.z);
                    }
                    sum[idx] += w;
                    C(idx, idx) += w * w;
                }
                break;
            }
            case PGL_BASIS_FUNC_SPLAT: {
                const uint32_t res = config.getResolution();
                const float sigma = config.getSplatSigma();
                const float alpha = -0.5f / (sigma * sigma);
                const float kernel_lb = std::exp(alpha);
                for (auto it = begin; it != end; ++it) {
                    pgl_vec2f p = dir_to_oct(it->reprojectedDirection);  // [0, 1]^2
                    // Splatting
                    // 3x3 Gaussian kernel
                    float bases[S];
                    for (uint8_t j = 0; j < S; ++j)
                        bases[j] = 0.0;

                    constexpr pgl_vec2i offsets[9] = {
                        {-1, -1}, {0, -1}, {+1, -1},
                        {-1,  0}, {0,  0}, {+1,  0},
                        {-1, +1}, {0, +1}, {+1, +1}
                    };

                    pgl_vec2i pi{
                        std::clamp((int)(p.x * res), 0, (int)res - 1),
                        std::clamp((int)(p.y * res), 0, (int)res - 1)
                    };  // {0, .., res-1}

                    // Dynamically compute kernel weights of each neighbor's center
                    float sumCoeff = 0;
                    for (int i = 0; i < 9; ++i) {
                        pgl_vec2i qi = {pi.x + offsets[i].x, pi.y + offsets[i].y};
                        // pgl_vec2f delta = {(float)(pi.x - qi.x), (float)(pi.y - qi.y)}; // old approach: static weights
                        pgl_vec2f delta = {p.x * res - (qi.x + 0.5f), p.y * res - (qi.y + 0.5f)};
                        float k = std::max(std::exp(alpha * (delta.x*delta.x + delta.y*delta.y)) - kernel_lb, 0.0f);
                        sumCoeff += k;

                        wrap_splat(qi.x, qi.y, res);
                        uint8_t j = pgl_pcg2d(qi.x, qi.y).first % S;  // hash to bin
                        bases[j] += k;
                    }

                    // Normalize weights
                    for (uint8_t j = 0; j < S; ++j) {
                        bases[j] /= sumCoeff;
                    }

                    // Splat the contribution to nearbying bins, each one with the statistical weight set as the kernelCoeff
                    for (uint8_t j = 0; j < S; ++j) {
                        float w = (multiplyCosine ? it->cosineTerm : 1) * it->weight;
                        sum[j] += bases[j] * w;
                        for (uint8_t h = j; h < S; ++h)
                            C(j, h) += bases[j] * bases[h] * w * w;
                    }
                }
                break;
            }
            case PGL_BASIS_FUNC_DON_PCG:
            case PGL_BASIS_FUNC_DON_XI: {
                const uint8_t log2_bin_count = (uint8_t) std::log2(S);
                const uint8_t octave_min = config.getOctaveMin(), octave_max = config.getOctaveMax();
                const float gamma = config.getDONGamma();
                if (config.basisType == PGL_BASIS_FUNC_DON_XI) {
                    if (S != (1 << log2_bin_count)) {
                        std::cerr << "Signature size must be a power of 2" << std::endl;
                        return;
                    }
                }
                for (auto it = begin; it != end; ++it) {
                    pgl_vec2f p = dir_to_oct(it->reprojectedDirection);  // [0, 1]^2
                    // Disjoint Octave Noise basis function
                    float bases[S];
                    for (uint8_t j = 0; j < S; ++j)
                        bases[j] = 0.0;
                    float normalizer = 0.0f;

                    // * Evaluates all basis functions at the given coordinate
                    // Iterate over all octaves
                    for (uint8_t k = octave_min; k <= octave_max; ++k) {
                        const uint32_t res = 1 << k;
                        // const float weight = pow(0.5, float(k)) / basis_normalizer;
                        const float weight = std::pow(gamma, float(k));
                        normalizer += weight;
                        // Discretize uv at the appropriate resolution
                        pgl_vec2f octave_uv = {p.x * float(res), p.y * float(res)};
                        uint32_t x00 = uint32_t(octave_uv.x), y00 = uint32_t(octave_uv.y);
                        // Generate offsets
                        uint32_t x01 = x00, y01 = y00 + 1;
                        uint32_t x10 = x00 + 1, y10 = y00;
                        uint32_t x11 = x00 + 1, y11 = y00 + 1;
                        // Apply wrapping to ensure continuity on the sphere domain
                        wrap(x00, y00, res);
                        wrap(x01, y01, res);
                        wrap(x10, y10, res);
                        wrap(x11, y11, res);
                        uint8_t h00;
                        uint8_t h01;
                        uint8_t h10;
                        uint8_t h11;
                        if (config.basisType == PGL_BASIS_FUNC_DON_PCG) {
                            h00 = pcg_3d(x00, y00, k) % S;
                            h01 = pcg_3d(x01, y01, k) % S;
                            h10 = pcg_3d(x10, y10, k) % S;
                            h11 = pcg_3d(x11, y11, k) % S;
                        } else {
                            // using Xi-seq for lower discrepancy and less clumping
                            h00 = get_bin(x00, y00, k, log2_bin_count);
                            h01 = get_bin(x01, y01, k, log2_bin_count);
                            h10 = get_bin(x10, y10, k, log2_bin_count);
                            h11 = get_bin(x11, y11, k, log2_bin_count);
                        }

                        for (uint8_t j = 0; j < S; ++j) {
                            // Determine whether this bin gets the sample
                            float M00 = (h00 == j) ? 1.0 : 0.0;
                            float M01 = (h01 == j) ? 1.0 : 0.0;
                            float M10 = (h10 == j) ? 1.0 : 0.0;
                            float M11 = (h11 == j) ? 1.0 : 0.0;
                            // Perform bilinear interpolation
                            float M0 = mix(M00, M01, fract(octave_uv.y));
                            float M1 = mix(M10, M11, fract(octave_uv.y));
                            float M = mix(M0, M1, fract(octave_uv.x));
                            // Accumulate into the result
                            bases[j] += weight * M;
                        }
                    }

                    // Contribute to all bins, each one attenuated with its basis function
                    for (uint8_t j = 0; j < S; ++j) {
                        float w = (multiplyCosine ? it->cosineTerm : 1) * it->weight;
                        sum[j] += bases[j] / normalizer * w;
                        for (uint8_t h = j; h < S; ++h)
                            C(j, h) += bases[j] / normalizer * bases[h] / normalizer * w * w;
                    }
                }

                break;
            }
            case PGL_BASIS_FUNC_LATITUDE: {
                const uint32_t res = config.getResolution();
                for (auto it = begin; it != end; ++it) {
                    pgl_vec2f p = dir_to_spherical(it->reprojectedDirection);  // [0, 1]^2
                    float u = fract(p.x * float(res));  // latitude

                    float bases[S];
                    for (uint8_t j = 0; j < S; ++j) {
                        float x = M_PI_2 * (float(S) * u - float(j));
                        float b = 0.0;
                        if ((-M_PI_2 <= x && x < M_PI_2) || (-M_PI_2 <= x - M_PI_2 * float(S) && x - M_PI_2 * float(S) < M_PI_2)) {
                            b = std::cos(x);
                            b *= b;
                        }
                        bases[j] = b;
                    }

                    for (uint8_t j = 0; j < S; ++j) {
                        float w = (multiplyCosine ? it->cosineTerm : 1) * it->weight;
                        sum[j] += bases[j] * w;
                        for (uint8_t h = j; h < S; ++h)
                            C(j, h) += bases[j] * bases[h] * w * w;
                    }
                }
                break;
            }
            case PGL_BASIS_FUNC_LONGITUDE: {
                const uint32_t res = config.getResolution();
                for (auto it = begin; it != end; ++it) {
                    pgl_vec2f p = dir_to_spherical(it->reprojectedDirection);  // [0, 1]^2
                    float v = fract(p.y * float(res));  // longitude

                    float bases[S];
                    for (uint8_t j = 0; j < S; ++j) {
                        float x = M_PI_2 * (float(S) * v - float(j));
                        float b = 0.0;
                        if ((-M_PI_2 <= x && x < M_PI_2) || (-M_PI_2 <= x - M_PI_2 * float(S) && x - M_PI_2 * float(S) < M_PI_2)) {
                            b = std::cos(x);
                            b *= b;
                        }
                        bases[j] = b;
                    }

                    for (uint8_t j = 0; j < S; ++j) {
                        float w = (multiplyCosine ? it->cosineTerm : 1) * it->weight;
                        sum[j] += bases[j] * w;
                        for (uint8_t h = j; h < S; ++h)
                            C(j, h) += bases[j] * bases[h] * w * w;
                    }
                }
                break;
            }
            case PGL_BASIS_FUNC_CHECKERBOARD: {
                const uint32_t res = config.getResolution();
                for (auto it = begin; it != end; ++it) {
                    uint8_t idx = get_signature_index_checkerboard(it->reprojectedDirection, res, S);
                    float w = it->weight;
                    if (multiplyCosine) {
                        w *= it->cosineTerm;
                    }
                    sum[idx] += w;
                    C(idx, idx) += w * w;
                }
                break;
            }

            default: throw std::runtime_error("Unknown basis type");
        }
    }

//     template<typename SampleIterator>
//     static void computeSampleBasisFunctions(SampleIterator begin, SampleIterator end, PGL_BASIS_FUNC_TYPE basisType) {
//         // Forwarding to the appropriate function
//         switch (basisType) {
//             case PGL_BASIS_FUNC_NN:
//                 return computeSampleBasisFunctions<PGL_BASIS_FUNC_NN>(begin, end);
//             case PGL_BASIS_FUNC_SPLAT:
//                 return computeSampleBasisFunctions<PGL_BASIS_FUNC_SPLAT>(begin, end);
//             case PGL_BASIS_FUNC_DON_PCG:
//                 return computeSampleBasisFunctions<PGL_BASIS_FUNC_DON_PCG>(begin, end);
//             case PGL_BASIS_FUNC_DON_XI:
//                 return computeSampleBasisFunctions<PGL_BASIS_FUNC_DON_XI>(begin, end);
//             default:
//                 throw std::runtime_error("Unknown contribution type");
//         }
//     }

//     template<PGL_BASIS_FUNC_TYPE basisType, typename SampleIterator>
//     static void computeSampleBasisFunctions(SampleIterator begin, SampleIterator end) {
// #ifdef OPENPGL_CACHE_BASIS_FUNCTIONS
//         const uint8_t S = g_opgl_signature_size;
//         const uint8_t log2_bin_count = (uint8_t) std::log2(S);
//         if constexpr(basisType == PGL_BASIS_FUNC_DON_XI) {
//             if (S != (1 << log2_bin_count)) {
//                 std::cerr << "Signature size must be a power of 2" << std::endl;
//                 return;
//             }
//         }
//         const uint8_t octave_min = g_opgl_octave_min, octave_max = g_opgl_octave_max;
//         // const float basis_normalizer = pow(2.0, 1.0 - float(octave_min)) - pow(0.5, float(octave_max));
//         const float alpha = -0.5f / (g_opgl_splat_sigma*g_opgl_splat_sigma);
//         const float gamma = g_opgl_octave_gamma;
//         const float kernel_lb = std::exp(alpha);

//         size_t N = std::distance(begin, end);
//         embree::parallel_for(N, [&](embree::range<size_t> r) {
//             for (size_t i = r.begin(); i < r.end(); ++i) {
//                 auto it = begin + i;
//                 if constexpr (basisType == PGL_BASIS_FUNC_DON_PCG || basisType == PGL_BASIS_FUNC_DON_XI) {
//                     pgl_vec2f p = dir_to_oct(it->reprojectedDirection); // [0, 1]^2
//                     // Disjoint Octave Noise basis function
//                     for (uint8_t j = 0; j < S; ++j)
//                         it->basisFunction[j] = 0.0;
//                     float normalizer = 0.0f;

//                     // * Evaluates all basis functions at the given coordinate
//                     // Iterate over all octaves
//                     for (uint8_t k = octave_min; k <= octave_max; ++k) {
//                         const uint32_t res = 1 << k;
//                         // const float weight = pow(0.5, float(k)) / basis_normalizer;
//                         const float weight = std::pow(gamma, float(k));
//                         normalizer += weight;
//                         // Discretize uv at the appropriate resolution
//                         pgl_vec2f octave_uv = {p.x * float(res), p.y * float(res)};
//                         uint32_t x00 = uint32_t(octave_uv.x), y00 = uint32_t(octave_uv.y);
//                         // Generate offsets
//                         uint32_t x01 = x00, y01 = y00 + 1;
//                         uint32_t x10 = x00 + 1, y10 = y00;
//                         uint32_t x11 = x00 + 1, y11 = y00 + 1;
//                         // Apply wrapping to ensure continuity on the sphere domain
//                         wrap(x00, y00, res);
//                         wrap(x01, y01, res);
//                         wrap(x10, y10, res);
//                         wrap(x11, y11, res);
//                         uint8_t h00;
//                         uint8_t h01;
//                         uint8_t h10;
//                         uint8_t h11;
//                         if constexpr (basisType == PGL_BASIS_FUNC_DON_PCG) {
//                             h00 = pcg_3d(x00, y00, k) % S;
//                             h01 = pcg_3d(x01, y01, k) % S;
//                             h10 = pcg_3d(x10, y10, k) % S;
//                             h11 = pcg_3d(x11, y11, k) % S;
//                         } else {
//                             // using Xi-seq for lower discrepancy and less clumping
//                             h00 = get_bin(x00, y00, k, log2_bin_count);
//                             h01 = get_bin(x01, y01, k, log2_bin_count);
//                             h10 = get_bin(x10, y10, k, log2_bin_count);
//                             h11 = get_bin(x11, y11, k, log2_bin_count);
//                         }

//                         for (uint8_t j = 0; j < S; ++j) {
//                             // Determine whether this bin gets the sample
//                             float M00 = (h00 == j) ? 1.0 : 0.0;
//                             float M01 = (h01 == j) ? 1.0 : 0.0;
//                             float M10 = (h10 == j) ? 1.0 : 0.0;
//                             float M11 = (h11 == j) ? 1.0 : 0.0;
//                             // Perform bilinear interpolation
//                             float M0 = mix(M00, M01, fract(octave_uv.y));
//                             float M1 = mix(M10, M11, fract(octave_uv.y));
//                             float M = mix(M0, M1, fract(octave_uv.x));
//                             // Accumulate into the result
//                             it->basisFunction[j] += weight * M;
//                         }
//                     }

//                     // Normalize
//                     for (uint8_t j = 0; j < S; ++j) {
//                         it->basisFunction[j] /= normalizer;
//                     }
//                 } else if constexpr (basisType == PGL_BASIS_FUNC_SPLAT) {
//                     pgl_vec2f p = dir_to_oct(it->reprojectedDirection); // [0, 1]^2
//                     // Splatting
//                     // 3x3 Gaussian kernel
//                     for (uint8_t j = 0; j < S; ++j)
//                         it->basisFunction[j] = 0.0;

//                     constexpr pgl_vec2i offsets[9] = {
//                         {-1, -1}, {0, -1}, {+1, -1},
//                         {-1, 0}, {0, 0}, {+1, 0},
//                         {-1, +1}, {0, +1}, {+1, +1}
//                     };

//                     pgl_vec2i pi{
//                         std::clamp((int) (p.x * g_opgl_octahedral_resolution), 0,
//                                    (int) g_opgl_octahedral_resolution - 1),
//                         std::clamp((int) (p.y * g_opgl_octahedral_resolution), 0,
//                                    (int) g_opgl_octahedral_resolution - 1)
//                     }; // {0, .., g_opgl_octahedral_resolution-1}

//                     // Dynamically compute kernel weights of each neighbor's center
//                     float sumCoeff = 0;
//                     for (int i = 0; i < 9; ++i) {
//                         pgl_vec2i qi = {pi.x + offsets[i].x, pi.y + offsets[i].y};
//                         // pgl_vec2f delta = {(float)(pi.x - qi.x), (float)(pi.y - qi.y)}; // old approach: static weights
//                         pgl_vec2f delta = {
//                             p.x * g_opgl_octahedral_resolution - (qi.x + 0.5f),
//                             p.y * g_opgl_octahedral_resolution - (qi.y + 0.5f)
//                         };
//                         float k = std::max(std::exp(alpha * (delta.x * delta.x + delta.y * delta.y)) - kernel_lb, 0.0f);
//                         sumCoeff += k;

//                         wrap_splat(qi.x, qi.y, g_opgl_octahedral_resolution);
//                         uint8_t j = pgl_pcg2d(qi.x, qi.y).first % S; // hash to bin
//                         it->basisFunction[j] += k;
//                     }

//                     // Normalize weights
//                     for (uint8_t j = 0; j < S; ++j) {
//                         it->basisFunction[j] /= sumCoeff;
//                     }
//                 } else if constexpr (basisType == PGL_BASIS_FUNC_NN) {
//                     // One-hot
//                     uint8_t idx = pgl_get_signature_index(it->reprojectedDirection);
//                     for (uint8_t j = 0; j < S; ++j) {
//                         it->basisFunction[j] = (j == idx) ? 1.0 : 0.0;
//                     }
//                 } else {
//                     throw std::runtime_error("Unknown contribution type");
//                 }
//             }
//         });
// #else
//         std::cerr << "Optimized path not available without OPENPGL_CACHE_BASIS_FUNCTIONS" << std::endl;
// #endif
//     }

    void addZeroSamples(size_t numZeroSamples) {
        numSamples += (float) numZeroSamples;
    }

    float getNumSamples() const {
        return numSamples;
    }

    float getFluence() const {
        float tot = 0;
        for (int i = 0; i < sum.size(); ++i)
            tot += sum[i];
        return tot / numSamples;
    }

    float getFluenceStd() const {
        float mu = getFluence();
        float m2 = 0;
        for (int i = 0; i < sum.size(); ++i)
            for (int j = 0; j < sum.size(); ++j)
                m2 += C(i, j);  // the sum of all comomentum entries is the second momentum of fluence
        float var = m2 / numSamples - mu * mu;
        return std::sqrt(var / numSamples);
    }

    float getMean(uint8_t idx) const {
        return sum[idx] / numSamples;
    }

    // N-sample std
    float getStd(uint8_t idx) const {
        return std::sqrt(getVariance(idx));
    }
    
    float getOneSampleStd(uint8_t idx) const {
        return std::sqrt(getOneSampleVariance(idx));
    }

    // N-sample variance
    float getVariance(uint8_t idx) const {
        return getOneSampleVariance(idx) / numSamples;
    }

    float getOneSampleVariance(uint8_t idx) const {
        return C(idx, idx) / numSamples - sum[idx] * sum[idx] / (numSamples * numSamples);  // assuming no covariance, biased
    }

    float getCovariance(uint8_t i, uint8_t j) const {
        float oneSampleCov = C(i, j) / numSamples - sum[i] * sum[j] / (numSamples * numSamples);
        return oneSampleCov / numSamples;  // N-sample covariance
    }

    void decay(float alpha) {
        for (int i = 0; i < sum.size(); i++)
            sum[i] *= alpha;
        for (int i = 0; i < com.size(); i++)
            com[i] *= alpha;
        numSamples *= alpha;
    }

    explicit operator PGLDirectionalSignature() const {
        PGLDirectionalSignature signature;
        for (uint8_t i = 0; i < sum.size(); i++) {
            signature.signature[i] = getMean(i);
            signature.std[i] = getStd(i);
        }
        signature.numSamples = numSamples;
        signature.S = sum.size();
        return signature;
    }

    // L2 distance between two signatures
    // Assuming a is parent, b is child
    static float getDistanceL2(const Signature &a, const Signature &b, float stdMultiplier) {
        float sum = 0;
        uint8_t S = b.sum.size();
        if (S == 1) {
            float ai = a.getFluence(), bi = b.getMean(0);
            float a_std = stdMultiplier * a.getFluenceStd(), b_std = stdMultiplier * b.getStd(0);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            float diff = 0;
            if (ai - a_std > bi + b_std)
                diff = ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                diff = bi - ai - a_std - b_std;
            sum += diff * diff;
        } else for (uint8_t i = 0; i < S; i++) {
            float ai = a.getMean(i), bi = b.getMean(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            float diff = 0;
            if (ai - a_std > bi + b_std)
                diff = ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                diff = bi - ai - a_std - b_std;
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // L1 distance between two signatures
    static float getDistanceL1(const Signature &a, const Signature &b, float stdMultiplier) {
        float sum = 0;
        uint8_t S = b.sum.size();
        if (S == 1) {
            float ai = a.getFluence(), bi = b.getMean(0);
            float a_std = stdMultiplier * a.getFluenceStd(), b_std = stdMultiplier * b.getStd(0);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                sum += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                sum += bi - ai - a_std - b_std;
        } else for (uint8_t i = 0; i < S; i++) {
            float ai = a.getMean(i), bi = b.getMean(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                sum += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                sum += bi - ai - a_std - b_std;
        }
        return sum;
    }

    // SMAPE: symmetric mean absolute percentage error, lookahead level can be inferred from b's size
    // Has a range of [0, 2]
    static float getDistanceSMAPE(const Signature &a, const Signature &b, float stdMultiplier) {
        float num = 0, denom = 0;
        uint8_t S = b.sum.size();
        if (S == 1) {
            float ai = a.getFluence(), bi = b.getMean(0);
            float a_std = stdMultiplier * a.getFluenceStd(), b_std = stdMultiplier * b.getStd(0);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                num += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                num += bi - ai - a_std - b_std;
            denom += ai;
        } else for (uint8_t i = 0; i < S; i++) {
            float ai = a.getMean(i), bi = b.getMean(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                num += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                num += bi - ai - a_std - b_std;
            denom += ai;
        }
        return a.numSamples == 0 ? 0 : num / denom;
    }

    // Assuming b is parent
    static float getOneSampleT(const Signature &a, const Signature &b) {
        // OPENPGL_ASSERT(g_opgl_signature_size == 1);
        float num = a.getMean(0) - b.getMean(0);
        // float denom = b.getStd(0);
        float denom = a.getStd(0);
        return denom == 0 ? 0 : num / denom;
    }

    static float getWelchT(const Signature &a, const Signature &b, float eps) {
        // OPENPGL_ASSERT(g_opgl_signature_size == 1);
        float num = a.getMean(0) - b.getMean(0);
        float denom = std::sqrt(a.getVariance(0) + b.getVariance(0)) + eps;
        return denom == 0 ? 0 : num / denom;
    }

    // In some cases, a is assumed to be the *parent* region
    static float getDistance(const Signature &a, const Signature &b, float stdMultiplier) {
        return getDistanceSMAPE(a, b, stdMultiplier);
    }

    // Only works for one bin
    static float getSufficientCriterionStatistics(const Signature &A, const Signature &B, float T) {
        // D is A - B
        float D_numSamples = A.numSamples - B.numSamples;
        float D_mean = (A.sum[0] - B.sum[0]) / D_numSamples;
        float D_s2 = ((A.com[0] - B.com[0]) / D_numSamples - D_mean * D_mean) / D_numSamples;
        float B_mean = B.getMean(0);
        float B_s2 = B.getOneSampleVariance(0) / B.numSamples;

        float wB_n = +(1-T) * B.numSamples - A.numSamples, wD_n = +(1-T) * D_numSamples;
        float wB_p = -(1+T) * B.numSamples + A.numSamples, wD_p = -(1+T) * D_numSamples;

        float criterion_n = (wB_n * B_mean + wD_n * D_mean) / std::sqrt(wB_n*wB_n * B_s2 + wD_n*wD_n * D_s2);
        float criterion_p = (wB_p * B_mean + wD_p * D_mean) / std::sqrt(wB_p*wB_p * B_s2 + wD_p*wD_p * D_s2);
        // return std::max( Phi(criterion_n) + Phi(criterion_p) );
        return std::max(Phi(criterion_n), Phi(criterion_p));
        // return std::max(criterion_n, criterion_p);
    }

    static float getDistanceTTest(const Signature &a, const Signature &b, float stdMultiplier, float tvalueThreshold) {
        float num = 0, denom = 0;
        uint8_t S = b.sum.size();
        if (S == 1) {
            float ai = a.getFluence(), bi = b.getMean(0);
            float a_std = stdMultiplier * a.getFluenceStd(), b_std = stdMultiplier * b.getStd(0);
            float sigma = std::sqrt(a.getFluenceStd() * a.getFluenceStd() + b.getVariance(0));
            float t = sigma == 0 ? 0 : (ai - bi) / sigma;
            if (std::abs(t) > tvalueThreshold) {
                // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
                if (ai - a_std > bi + b_std)
                    num += ai - bi - a_std - b_std;
                else if (ai + a_std < bi - b_std)
                    num += bi - ai - a_std - b_std;
            }
            denom += ai + bi;
        } else for (uint8_t i = 0; i < S; i++) {
            float ai = a.getMean(i), bi = b.getMean(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            float sigma = std::sqrt(a.getVariance(i) + b.getVariance(i));
            float t = sigma == 0 ? 0 : (ai - bi) / sigma;
            if (std::abs(t) > tvalueThreshold) {
                // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
                if (ai - a_std > bi + b_std)
                    num += ai - bi - a_std - b_std;
                else if (ai + a_std < bi - b_std)
                    num += bi - ai - a_std - b_std;
            }
            denom += ai + bi;
        }
        return denom == 0 ? 0 : 2.0f * num / denom;
    }

    // Estimates the split probability as through Monte-Carlo simulations, assuming bin values follow multivariate Gaussian distribution
    static float getSplitProbaMC(const Signature &A, const Signature &B, int numSimSamples, float T) {
        const int S = B.sum.size();
        if (S == 1) return getSufficientCriterionStatistics(A, B, T);  // One bin case, we have a closed formula :)
        // D = A \ B
        Signature D = A - B;
        const float fluence = A.getFluence();

        // Mean vectors
        float muB[S], muD[S];
        for (int j = 0; j < S; ++j) {
            muB[j] = B.getMean(j);
            muD[j] = D.getMean(j);
        }

        // Covariance matrices
        float covB[S][S], covD[S][S];
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                covB[i][j] = B.getCovariance(i, j);
                covD[i][j] = D.getCovariance(i, j);
            }

        size_t seed = A.numSamples;  // TODO: better ways?
        MultivariateNormalSampler samplerB(S, muB, &covB[0][0], seed);
        MultivariateNormalSampler samplerD(S, muD, &covD[0][0], seed * seed);

        // Run Monte-Carlo simulation
        int positiveCount = 0;
        for (int i = 0; i < numSimSamples; ++i) {
            float XB[S], XD[S];
            samplerB.draw(XB);
            samplerD.draw(XD);

            float energy = 0;
            for (int j = 0; j < S; ++j) {
                energy += std::abs(XD[j] - XB[j]);
            }
            energy *= D.numSamples / A.numSamples / fluence;
            
            if (energy > T) ++positiveCount;
        }

        return (float) positiveCount / (float) numSimSamples;
    }

    static inline float Phi(float x) {
        return 0.5f * (std::erf(x / std::sqrt(2)) + 1);
    }

    float getRisk() const {
        // maximum value of std / mean per bin
        float maxRisk = 0;
        for (uint8_t i = 0; i < sum.size(); i++) {
            float mean = getMean(i);
            if (mean == 0) continue;
            float std = getStd(i);
            maxRisk = std::max(maxRisk, std / mean);
        }
        return maxRisk;
    }

    void serialize(std::ostream &stream) const {
        int size = sum.size();
        stream.write(reinterpret_cast<const char *>(&size), sizeof(size));
        for (int i = 0; i < size; ++i) {
            stream.write(reinterpret_cast<const char *>(&sum[i]), sizeof(sum[i]));
        }
        size = ((size + 1) * size) >> 1;
        for (int i = 0; i < size; ++i) {
            stream.write(reinterpret_cast<const char *>(&com[i]), sizeof(com[i]));
        }
        stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(numSamples));
    }

    void deserialize(std::istream &stream) {
        int size;
        stream.read(reinterpret_cast<char *>(&size), sizeof(size));
        sum.resize(size);
        for (int i = 0; i < size; ++i) {
            stream.read(reinterpret_cast<char *>(&sum[i]), sizeof(sum[i]));
        }
        size = ((size + 1) * size) >> 1;
        com.resize(size);
        for (int i = 0; i < size; ++i) {
            stream.read(reinterpret_cast<char *>(&com[i]), sizeof(com[i]));
        }
        stream.read(reinterpret_cast<char *>(&numSamples), sizeof(numSamples));
    }
};

} // namespace openpgl
