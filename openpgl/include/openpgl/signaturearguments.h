//
// Created by fengshi on 4/29/25.
//

#pragma once

#include "common.h"
#include "types.h"
#include <cassert>

struct SignatureArguments {
    PGL_BASIS_FUNC_TYPE basisType : 3 {PGL_BASIS_FUNC_NN};
    uint8_t numBins : 8 {8};   // maximum 254
    uint32_t param0 : 21 {0};  // resolution, {octave_min, octave_max}, etc
    uint32_t param1 {0};   // gamma, splat_sigma

    SignatureArguments() {
        setType(PGL_BASIS_FUNC_NN);
    }

    inline bool operator==(const SignatureArguments &b) const {
        return basisType == b.basisType && numBins == b.numBins &&
               param0 == b.param0 && param1 == b.param1;
    }

    inline bool is(PGL_BASIS_FUNC_TYPE type) const {
        return basisType == type;
    }

    inline void setType(PGL_BASIS_FUNC_TYPE type) {
        basisType = type;
        switch (type) {
            case PGL_BASIS_FUNC_NN:
                setResolution(64);
                break;
            case PGL_BASIS_FUNC_SPLAT:
                setResolution(64);
                setSplatSigma(1.0f);
                break;
            case PGL_BASIS_FUNC_DON_PCG:
            case PGL_BASIS_FUNC_DON_XI:
                setOctaveMin(3);
                setOctaveMax(7);
                setDONGamma(0.5f);
                break;
            case PGL_BASIS_FUNC_LATITUDE:
                setResolution(2);
                break;
            case PGL_BASIS_FUNC_LONGITUDE:
                setResolution(4);
                break;
            case PGL_BASIS_FUNC_CHECKERBOARD:
                setResolution(64);
                break;
        }
    }

    inline uint32_t getResolution() const {
        assert(is(PGL_BASIS_FUNC_NN) || is(PGL_BASIS_FUNC_SPLAT) || is(PGL_BASIS_FUNC_LATITUDE) || is(PGL_BASIS_FUNC_LONGITUDE) || is(PGL_BASIS_FUNC_CHECKERBOARD));
        return param0;
    }

    inline void setResolution(uint32_t resolution) {
        assert(is(PGL_BASIS_FUNC_NN) || is(PGL_BASIS_FUNC_SPLAT) || is(PGL_BASIS_FUNC_LATITUDE) || is(PGL_BASIS_FUNC_LONGITUDE) || is(PGL_BASIS_FUNC_CHECKERBOARD));
        param0 = resolution;
    }

    inline uint8_t getOctaveMin() const {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        return (param0 >> 8) & 0xFF;
    }

    inline void setOctaveMin(uint8_t octaveMin) {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        param0 = (param0 & 0xFFFF00FF) | (uint32_t(octaveMin) << 8);
    }

    inline uint8_t getOctaveMax() const {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        return param0 & 0xFF;
    }

    inline void setOctaveMax(uint8_t octaveMax) {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        param0 = (param0 & 0xFFFFFF00) | uint32_t(octaveMax);
    }

    inline float getSplatSigma() const {
        assert(is(PGL_BASIS_FUNC_SPLAT));
        return reinterpret_cast<const float &>(param1);
    }

    inline void setSplatSigma(float sigma) {
        assert(is(PGL_BASIS_FUNC_SPLAT));
        param1 = reinterpret_cast<const uint32_t &>(sigma);
    }

    inline float getDONGamma() const {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        return reinterpret_cast<const float &>(param1);
    }

    inline void setDONGamma(float gamma) {
        assert(is(PGL_BASIS_FUNC_DON_PCG) || is(PGL_BASIS_FUNC_DON_XI));
        param1 = reinterpret_cast<const uint32_t &>(gamma);
    }
};
