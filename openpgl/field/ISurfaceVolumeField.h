// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleDataStorage.h"
#include "../directional/ISurfaceSamplingDistribution.h"
#include "../directional/IVolumeSamplingDistribution.h"
#include "../include/openpgl/regionstatistics.h"

namespace openpgl
{

struct FieldStatistics;

struct ISurfaceVolumeField
{
    using SampleContainer = SampleDataStorage::SampleContainer;

    virtual ~ISurfaceVolumeField(){};

    virtual ISurfaceSamplingDistribution *newSurfaceSamplingDistribution() const = 0;

    virtual bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual bool initSurfaceSamplingDistributionFine(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual IVolumeSamplingDistribution *newVolumeSamplingDistribution() const = 0;

    virtual bool initVolumeSamplingDistribution(IVolumeSamplingDistribution *volumeSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual void setSceneBounds(const openpgl::BBox &sceneBounds) = 0;

    virtual openpgl::BBox getSceneBounds() const = 0;

    virtual void updateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) = 0;

    virtual void evaluateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) = 0;

    virtual void updateFieldSurface(SampleContainer &samplesSurface) = 0;

    virtual void updateFieldVolume(SampleContainer &samplesVolume) = 0;

    virtual void clearCEStatistics() = 0;

    virtual void clearSignatures() = 0;

    virtual void updateSubdivConfig(const PGLKDTreeArguments &cfg) = 0;

    virtual void loadSubdivConfig(PGLKDTreeArguments &cfg) const = 0;

    virtual void resetField() = 0;

    virtual PGL_SPATIAL_STRUCTURE_TYPE getSpatialStructureType() const = 0;

    virtual PGL_DIRECTIONAL_DISTRIBUTION_TYPE getDirectionalDistributionType() const = 0;

    virtual size_t getIteration() const = 0;

    virtual void serialize(std::ostream &os) const = 0;

    virtual void deserialize(std::istream &is) = 0;

    virtual bool validate(const bool checkSurface, const bool checkVolume) const = 0;

    virtual void storeToFile(const std::string fieldFileName) const = 0;

    virtual bool operator==(const ISurfaceVolumeField *b) const = 0;

    virtual FieldStatistics *getSurfaceStatistics() const = 0;

    virtual FieldStatistics *getVolumeStatistics() const = 0;

    virtual size_t getRegionCountSurface() const = 0;

    virtual size_t getLeafCountSurface() const = 0;

    virtual PGLRegionStatistics getRegionStatsSurface(uint32_t id) const = 0;

    // Deprecated
    virtual std::pair<PGLRegionStatistics, PGLRegionStatistics> getCoarseFineRegionStatsSurface(const Point3 &position) const = 0;

    virtual PGLDirectionalSignature getDirectionalSignature(const Point3 &pos) const = 0;

    virtual std::pair<PGLDirectionalSignature, PGLDirectionalSignature> getLRDirectionalSignatures(const openpgl::Point3 &pos, uint8_t dim) const = 0;

    virtual uint8_t getDirectionalSignatureBestDim(const openpgl::Point3 &pos) const = 0;
};
}  // namespace openpgl
