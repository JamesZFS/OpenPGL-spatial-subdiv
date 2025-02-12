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
    void addSamples(SampleIterator begin, SampleIterator end, bool multiplyCosine, bool splat) {
        if (splat) {
            if (multiplyCosine) addSamples<true, true>(begin, end);
            else addSamples<false, true>(begin, end);
        } else {
            if (multiplyCosine) addSamples<true, false>(begin, end);
            else addSamples<false, false>(begin, end);
        }
    }

    // Convert the direction into [0, 1] representation on the octahedral map
    pgl_vec2f dir_to_oct(const pgl_direction &dir) {
        auto uv = pgl_vec2f(dir);  // [-1, 1]
        return {uv.x * 0.5f + 0.5f, uv.y * 0.5f + 0.5f};
    }

    template<bool multiplyCosine, bool splat, typename SampleIterator>
    void addSamples(SampleIterator begin, SampleIterator end) {
        for (auto it = begin; it != end; ++it) {
            if constexpr(splat) {
                // 3x3 Gaussian kernel
                float kernelCoeffs[9];
                uint8_t binIndices[9];
                constexpr float alpha = -std::log(2.f);  // alpha = -1/(2sigma^2)

                constexpr pgl_vec2i offsets[9] = {
                    {-1, -1}, {0, -1}, {+1, -1},
                    {-1,  0}, {0,  0}, {+1,  0},
                    {-1, +1}, {0, +1}, {+1, +1}
                };

                pgl_vec2f p = dir_to_oct(it->reprojectedDirection);  // [0, 1]^2
                pgl_vec2i pi{
                    std::clamp((int)(p.x * g_opgl_octahedral_resolution), 0, (int)g_opgl_octahedral_resolution - 1),
                    std::clamp((int)(p.y * g_opgl_octahedral_resolution), 0, (int)g_opgl_octahedral_resolution - 1)
                };  // {0, .., g_opgl_octahedral_resolution-1}

                // Dynamically compute kernel weights of each neighbor's center
                float sumCoeff = 0;
                for (int i = 0; i < 9; ++i) {
                    pgl_vec2i qi = {
                        std::clamp(pi.x + offsets[i].x, 0, (int)g_opgl_octahedral_resolution - 1), 
                        std::clamp(pi.y + offsets[i].y, 0, (int)g_opgl_octahedral_resolution - 1)
                    };
                    binIndices[i] = pgl_pcg2d(qi.x, qi.y).first % g_opgl_signature_size;  // hash to bin
                    
                    // pgl_vec2f delta = {(float)(pi.x - qi.x), (float)(pi.y - qi.y)}; // old approach: static weights
                    pgl_vec2f delta = {p.x * g_opgl_octahedral_resolution - (qi.x + 0.5f), p.y * g_opgl_octahedral_resolution - (qi.y + 0.5f)};
                    kernelCoeffs[i] = std::exp(alpha * (delta.x*delta.x + delta.y*delta.y));
                    sumCoeff += kernelCoeffs[i];
                }

                // Normalize weights
                for (int i = 0; i < 9; ++i) {
                    kernelCoeffs[i] /= sumCoeff;
                }

                // Splat the contribution to nearbying bins, each one with the statistical weight set as the kernelCoeff
                for (int i = 0; i < 9; ++i) {
                    float k = kernelCoeffs[i];
                    float w = it->weight;
                    if constexpr(multiplyCosine) w *= it->cosineTerm;
                    uint8_t idx = binIndices[i];

                    sum[idx] += k * w;
                    m2[idx] += k * w * w;
                    numSamples2 += k * k;  // * Special
                }
                ++numSamples;
            } else {
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
