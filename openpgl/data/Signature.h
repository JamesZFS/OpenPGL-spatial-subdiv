// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl {

struct Signature  // Directional signature
{
    float sum[PGL_SIGNATURE_MAX_SIZE] = {};  // sum of weights in that bin
    float m2[PGL_SIGNATURE_MAX_SIZE] = {};  // sum of squared weights in that bin
    float numSamples = 0;  // number of samples in all bins, or sum of sample weights
    float numSamples2 = 0;  // sum of sample weights^2, only differs from numSamples after decay

    Signature() = default;

    explicit Signature(const PGLDirectionalSignature &s) {
        numSamples = numSamples2 = s.numSamples;
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            sum[i] = s.signature[i] * numSamples;
            float varNSample = s.std[i] * s.std[i];
            float varOneSample = numSamples * varNSample;
            m2[i] = numSamples * (varOneSample + s.signature[i] * s.signature[i]);
        }
    }

    void clear() {
        memset(sum, 0, sizeof(sum));
        memset(m2, 0, sizeof(m2));
        numSamples = 0;
        numSamples2 = 0;
    }

    template<typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end, bool multiplyCosine, PGL_SPATIAL_CONTRIB_TYPE contribType) {
        // Forwarding to the appropriate function
        if (multiplyCosine) {
            switch (contribType) {
                case PGL_SPATIAL_CONTRIB_NN:
                    return addSamples<true, PGL_SPATIAL_CONTRIB_NN>(begin, end);
                case PGL_SPATIAL_CONTRIB_SPLAT:
                    return addSamples<true, PGL_SPATIAL_CONTRIB_SPLAT>(begin, end);
                case PGL_SPATIAL_CONTRIB_BASIS:
                    return addSamples<true, PGL_SPATIAL_CONTRIB_BASIS>(begin, end);
                default:
                    throw std::runtime_error("Unknown contribution type");
            }
        } else {
            switch (contribType) {
                case PGL_SPATIAL_CONTRIB_NN:
                    return addSamples<false, PGL_SPATIAL_CONTRIB_NN>(begin, end);
                case PGL_SPATIAL_CONTRIB_SPLAT:
                    return addSamples<false, PGL_SPATIAL_CONTRIB_SPLAT>(begin, end);
                case PGL_SPATIAL_CONTRIB_BASIS:
                    return addSamples<false, PGL_SPATIAL_CONTRIB_BASIS>(begin, end);
                default:
                    throw std::runtime_error("Unknown contribution type");
            }
        }
    }

    // Convert the direction into [0, 1] representation on the octahedral map
    pgl_vec2f dir_to_oct(const pgl_direction &dir) {
        auto uv = pgl_vec2f(dir);  // [-1, 1]
        return {uv.x * 0.5f + 0.5f, uv.y * 0.5f + 0.5f};
    }

    //A pseudorandom number generator with a seed consisting of 3 uints
    static uint32_t pcg_3d(uint32_t x, uint32_t y, uint32_t z) {
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

    template<bool multiplyCosine, PGL_SPATIAL_CONTRIB_TYPE contribType, typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        const uint8_t S = g_opgl_signature_size;
        const uint8_t octave_min = g_opgl_octave_min, octave_max = g_opgl_octave_max;
        // const float basis_normalizer = pow(2.0, 1.0 - float(octave_min)) - pow(0.5, float(octave_max));
        const float alpha = -0.5f / (g_opgl_splat_sigma*g_opgl_splat_sigma);
        const float gamma = g_opgl_octave_gamma;
        const float kernel_lb = std::exp(alpha);

        for (auto it = begin; it != end; ++it) {
            if constexpr(contribType == PGL_SPATIAL_CONTRIB_BASIS) {
                pgl_vec2f p = dir_to_oct(it->reprojectedDirection);  // [0, 1]^2
                // Disjoint Octave Noise basis function
                float basisFunctions[PGL_SIGNATURE_MAX_SIZE];
                for (uint8_t j = 0; j < S; ++j)
                    basisFunctions[j] = 0.0;
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
                    uint8_t h00 = pcg_3d(x00, y00, k) % S;
                    uint8_t h01 = pcg_3d(x01, y01, k) % S;
                    uint8_t h10 = pcg_3d(x10, y10, k) % S;
                    uint8_t h11 = pcg_3d(x11, y11, k) % S;

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
                        basisFunctions[j] += weight * M;
                    }
                }

                // Contribute to all bins, each one attenuated with its basis function
                for (uint8_t j = 0; j < S; ++j) {
                    float w = basisFunctions[j] / normalizer * it->weight;
                    if constexpr(multiplyCosine) w *= it->cosineTerm;
                    sum[j] += w;
                    m2[j] += w * w;
                }
                ++numSamples;
                ++numSamples2;
            } else if constexpr(contribType == PGL_SPATIAL_CONTRIB_SPLAT) {
                pgl_vec2f p = dir_to_oct(it->reprojectedDirection);  // [0, 1]^2
                // Splatting
                // 3x3 Gaussian kernel
                float maskingFunctions[PGL_SIGNATURE_MAX_SIZE];
                for (uint8_t j = 0; j < S; ++j)
                    maskingFunctions[j] = 0.0;

                constexpr pgl_vec2i offsets[9] = {
                    {-1, -1}, {0, -1}, {+1, -1},
                    {-1,  0}, {0,  0}, {+1,  0},
                    {-1, +1}, {0, +1}, {+1, +1}
                };

                pgl_vec2i pi{
                    std::clamp((int)(p.x * g_opgl_octahedral_resolution), 0, (int)g_opgl_octahedral_resolution - 1),
                    std::clamp((int)(p.y * g_opgl_octahedral_resolution), 0, (int)g_opgl_octahedral_resolution - 1)
                };  // {0, .., g_opgl_octahedral_resolution-1}

                // Dynamically compute kernel weights of each neighbor's center
                float sumCoeff = 0;
                for (int i = 0; i < 9; ++i) {
                    pgl_vec2i qi = {pi.x + offsets[i].x, pi.y + offsets[i].y};
                    // pgl_vec2f delta = {(float)(pi.x - qi.x), (float)(pi.y - qi.y)}; // old approach: static weights
                    pgl_vec2f delta = {p.x * g_opgl_octahedral_resolution - (qi.x + 0.5f), p.y * g_opgl_octahedral_resolution - (qi.y + 0.5f)};
                    float k = std::max(std::exp(alpha * (delta.x*delta.x + delta.y*delta.y)) - kernel_lb, 0.0f);
                    sumCoeff += k;
                    
                    wrap_splat(qi.x, qi.y, g_opgl_octahedral_resolution);
                    uint8_t j = pgl_pcg2d(qi.x, qi.y).first % S;  // hash to bin
                    maskingFunctions[j] += k;
                }

                // Normalize weights
                for (uint8_t j = 0; j < S; ++j) {
                    maskingFunctions[j] /= sumCoeff;
                }

                // Splat the contribution to nearbying bins, each one with the statistical weight set as the kernelCoeff
                for (uint8_t j = 0; j < S; ++j) {
                    float w = maskingFunctions[j] * it->weight;
                    if constexpr(multiplyCosine) w *= it->cosineTerm;
                    sum[j] += w;
                    m2[j] += w * w;
                }
                ++numSamples;
                ++numSamples2;
            } else if constexpr(contribType == PGL_SPATIAL_CONTRIB_NN) {
                // uint8_t idx = pgl_get_signature_index(it->reprojectedDirection);
                uint8_t idx = it->binIndex;
                float w = it->weight;
                if constexpr(multiplyCosine) {
                    w *= it->cosineTerm;
                    // pgl_vec3f dir = it->reprojectedDirection;
                    // pgl_vec3f normal = it->normal;
                    // w *= std::max(0.0f, dir.x * normal.x + dir.y * normal.y + dir.z * normal.z);
                }
                sum[idx] += w;
                m2[idx] += w * w;
                ++numSamples;
                ++numSamples2;
            } else {
                throw std::runtime_error("Unknown contribution type");
            }
        }
    }

    void addZeroSamples(size_t numZeroSamples) {
        numSamples += (float) numZeroSamples;
        numSamples2 += (float) numZeroSamples;
    }

    float getNumSamples() const {
        return numSamples;
    }

    float getEntry(uint8_t idx) const {
        return sum[idx] / numSamples;
    }

    float getStd(uint8_t idx) const {
        // return m2[idx] / numSamples - sum[idx] * sum[idx] / (numSamples * numSamples);   // one sample, biased
        float varOneSample = m2[idx] / numSamples - sum[idx] * sum[idx] / (numSamples * numSamples);  // assuming no covariance, biased
        // float varNSample = numSamples2 / (numSamples * numSamples) * varOneSample;
        float varNSample = numSamples2 / (numSamples * numSamples) * varOneSample;
        return std::sqrt(varNSample);
    }

    void decay(float alpha) {
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            sum[i] *= alpha;
            m2[i] *= alpha;
        }
        numSamples *= alpha;
        numSamples2 *= alpha * alpha;
    }

    explicit operator PGLDirectionalSignature() const {
        PGLDirectionalSignature signature;
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            signature.signature[i] = getEntry(i);
            signature.std[i] = getStd(i);
        }
        signature.numSamples = numSamples;
        return signature;
    }

    // L2 distance between two signatures
    static float getDistanceL2(const Signature &a, const Signature &b, float stdMultiplier) {
        float sum = 0;
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            float ai = a.getEntry(i), bi = b.getEntry(i);
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
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            float ai = a.getEntry(i), bi = b.getEntry(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                sum += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                sum += bi - ai - a_std - b_std;
        }
        return sum;
    }

    // SMAPE: symmetric mean absolute percentage error
    // Has a range of [0, 2]
    static float getDistanceSMAPE(const Signature &a, const Signature &b, float stdMultiplier) {
        float num = 0, denom = 0;
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            float ai = a.getEntry(i), bi = b.getEntry(i);
            float a_std = stdMultiplier * a.getStd(i), b_std = stdMultiplier * b.getStd(i);
            // accumulate when interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap
            if (ai - a_std > bi + b_std)
                num += ai - bi - a_std - b_std;
            else if (ai + a_std < bi - b_std)
                num += bi - ai - a_std - b_std;
            denom += ai + bi;
        }
        return denom == 0 ? 0 : 2.0f * num / denom;
    }

    static float getDistance(const Signature &a, const Signature &b, float stdMultiplier) {
        return getDistanceSMAPE(a, b, stdMultiplier);
    }

    static bool differsSignificantly(const Signature &a, const Signature &b, float threshold) {
        for (uint8_t i = 0; i < g_opgl_signature_size; i++) {
            float ai = a.getEntry(i), bi = b.getEntry(i);
            float a_std = a.getStd(i), b_std = b.getStd(i);
            // if interval [ai-a_std, ai+a_std] and [bi-b_std, bi+b_std] not overlap and SMAPE(ai, bi) > threshold => split!
            // if ((ai - a_std > bi + b_std || ai + a_std < bi - b_std) &&
            //     2.0f * std::abs(ai - bi) / (ai + bi) > threshold)
            //     return true;
            if ((ai - a_std > bi + b_std && 2.0f * (ai - bi - a_std - b_std) / (ai + bi) > threshold) ||
                (ai + a_std < bi - b_std && 2.0f * (bi - ai - a_std - b_std) / (ai + bi) > threshold))
                return true;
        }
        return false;
    }

    void serialize(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(sum), sizeof(sum));
        stream.write(reinterpret_cast<const char *>(m2), sizeof(m2));
        stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(numSamples));
        stream.write(reinterpret_cast<const char *>(&numSamples2), sizeof(numSamples2));
    }

    void deserialize(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(sum), sizeof(sum));
        stream.read(reinterpret_cast<char *>(m2), sizeof(m2));
        stream.read(reinterpret_cast<char *>(&numSamples), sizeof(numSamples));
        stream.read(reinterpret_cast<char *>(&numSamples2), sizeof(numSamples2));
    }
};

} // namespace openpgl
