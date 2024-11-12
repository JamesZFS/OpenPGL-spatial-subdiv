#pragma once

#include "../openpgl_common.h"
#define COMPUTE_CE_STYLE 0  // 0: instantiating (parallaxed, cosined) guiding distributions at each sample location; 1: projecting samples to the center of the region

namespace openpgl
{
    struct DivergenceStatistics
    {
        float numSamples {0};
        
        // Biased estimation statistics
        float wSum{0.0f};  // sum of MC weight, Li / qs, or fluence estimator
        float eSum{0.0f};  // sum of unnormalized cross entropy -Li / qs * log(q)
        float dSum{0.0f};  // sum of chi2 divergence denominator (Li / qs)^2 qs / q

        inline void clear()
        {
            numSamples = 0.0f;
            wSum = 0.0f;
            eSum = 0.0f;
            dSum = 0.0f;
        }

        // MC weight (Li / qs), sampling pdf, and model pdf
        inline void addSample(float weight, float qs, float q)
        {
            if (q <= 0) return;
            numSamples++;
            wSum += weight;
            eSum += -weight * std::log(q);
            dSum += weight * weight * qs / q;
            // dSum += weight * weight;
        }

        inline void addZeroWeightSamples(size_t numSamples)
        {
            this->numSamples += (float) numSamples;
        }

        inline float getNumSamples() const
        {
            return numSamples;
        }

        inline float getFluence() const
        {
            return numSamples > 0 ? wSum / numSamples : 0.0f;
        }

        inline float getCE() const
        {
            return eSum / (wSum + 1e-3f);
        }

        inline float getChi2Div() const
        {
            // return numSamples * dSum / (wSum * wSum + 1e-3f);
            return numSamples * dSum / (wSum * wSum + 1e-3f) - 1;
        }

        inline void decay(float a)
        {
            numSamples *= a;
            wSum *= a;
            eSum *= a;
            dSum *= a;
        }

        DivergenceStatistics merge(const DivergenceStatistics &b) const
        {
            DivergenceStatistics result;
            result.numSamples = numSamples + b.numSamples;
            result.wSum = wSum + b.wSum;
            result.eSum = eSum + b.eSum;
            result.dSum = dSum + b.dSum;

            return result;
        }

        void serialize(std::ostream& stream) const
        {
            stream.write(reinterpret_cast<const char*>(&numSamples), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&wSum), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&eSum), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&dSum), sizeof(float));
        }

        void deserialize(std::istream& stream)
        {
            stream.read(reinterpret_cast<char*>(&numSamples), sizeof(float));
            stream.read(reinterpret_cast<char*>(&wSum), sizeof(float));
            stream.read(reinterpret_cast<char*>(&eSum), sizeof(float));
            stream.read(reinterpret_cast<char*>(&dSum), sizeof(float));
        }

    };

}