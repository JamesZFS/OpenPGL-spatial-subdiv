// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Field.h"
#include "FieldStatistics.h"
#include "ISurfaceVolumeField.h"

#define FIELD_FILE_HEADER_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_FIELD"

namespace openpgl
{

template <int Vecsize, class TDirectionalDistributionFactory, template <typename, typename, typename, typename> class TSpatialStructureBuilder, typename TSurfaceSamplingDistribution,
          typename TVolumeSamplingDistribution>
struct SurfaceVolumeField : public ISurfaceVolumeField
{
   private:
    using SurfaceFieldType = Field<Vecsize, TDirectionalDistributionFactory, TSpatialStructureBuilder, TSurfaceSamplingDistribution>;
    using VolumeFieldType = Field<Vecsize, TDirectionalDistributionFactory, TSpatialStructureBuilder, TVolumeSamplingDistribution>;
    using SampleContainer = SampleDataStorage::SampleContainer;

   public:
    using SurfaceSettings = typename SurfaceFieldType::Settings;
    using SurfaceRegionType = typename SurfaceFieldType::RegionType;
    using VolumeSettings = typename VolumeFieldType::Settings;
    using VolumeRegionType = typename VolumeFieldType::RegionType;

    using Settings = SurfaceSettings;
    using DirectionalDistribution = typename SurfaceFieldType::DirectionalDistribution;

   public:
    SurfaceVolumeField() = default;

    SurfaceVolumeField(const SurfaceSettings &settings) : m_surfaceField(settings), m_volumeField(reinterpret_cast<const VolumeSettings&>(settings))
    {
        m_surfaceField.setIsSurface(true);
        m_volumeField.setIsSurface(false);
    }

    ~SurfaceVolumeField() override
    {
        std::cout << "SurfaceVolumeField::updateField() took " << m_timeUpdateField << " ms in total" << std::endl;
    }

    ISurfaceSamplingDistribution *newSurfaceSamplingDistribution() const override
    {
        return new TSurfaceSamplingDistribution();
    }

    bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample) const override
    {
        return _initSurfaceSamplingDistribution(surfaceSamplingDistribution, position, sample);
    }

    bool _initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample) const
    {
        TSurfaceSamplingDistribution *_surfaceSamplingDistribution = (TSurfaceSamplingDistribution *)surfaceSamplingDistribution;
        uint32_t id = -1;
        const SurfaceRegionType *region;
        region = m_surfaceField.getRegion(position, sample, id);
        if (!region || !region->valid)
        {
            return false;
        }
        const DirectionalDistribution *distribution = &region->distribution;
        _surfaceSamplingDistribution->init(distribution, position);
        _surfaceSamplingDistribution->setId(id);
        _surfaceSamplingDistribution->setRegion(region);
        return true;
    }

    IVolumeSamplingDistribution *newVolumeSamplingDistribution() const override
    {
        return new TVolumeSamplingDistribution();
    }

    bool initVolumeSamplingDistribution(IVolumeSamplingDistribution *volumeSamplingDistribution, const Point3 &position, float *sample1D) const override
    {
        TVolumeSamplingDistribution *_volumeSamplingDistribution = (TVolumeSamplingDistribution *)volumeSamplingDistribution;
        uint32_t id = -1;
        const VolumeRegionType *region = m_volumeField.getRegion(position, sample1D, id);
        if (!region || !region->valid)
        {
            return false;
        }
        const DirectionalDistribution *distribution = region->getDistribution(position);
        _volumeSamplingDistribution->init(distribution, position);
        _volumeSamplingDistribution->setId(id);
        _volumeSamplingDistribution->setRegion(region);
        return true;
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds) override
    {
        openpgl::BBox scaledSceneBounds = sceneBounds;
        scaledSceneBounds.enlarge_by(1.01f);
        m_surfaceField.setSceneBounds(scaledSceneBounds);
        m_volumeField.setSceneBounds(scaledSceneBounds);
    }

    openpgl::BBox getSceneBounds() const override
    {
        openpgl::BBox sceneBounds = m_surfaceField.getSceneBounds();
        sceneBounds.extend(m_volumeField.getSceneBounds());
        return sceneBounds;
    }

    void updateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) override
    {
        Timer timer;
#if TBB_INTERFACE_VERSION < 12010
        // we need to initialize the task_scheduler in the context to avoid
        // asyncronous deconsrution of the implicit initialized tbb::arenas and tbb::streams
        tbb::task_scheduler_init anonymous;
#endif
        if (samplesSurface.samples.size() > 0)
        {
            if (!m_surfaceField.isInitialized())
            {
                m_surfaceField.buildField(samplesSurface);
            }
            else
            {
                m_surfaceField.updateField(samplesSurface);
            }
        }
        if (samplesVolume.samples.size() > 0)
        {
            if (!m_volumeField.isInitialized())
            {
                m_volumeField.buildField(samplesVolume);
            }
            else
            {
                m_volumeField.updateField(samplesVolume);
            }
        }
        m_iteration++;
        m_timeUpdateField += timer.elapsed() * 1e-3f;
    }

    void evaluateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) override
    {
        if (samplesSurface.samples.size() > 0)
        {
            if (m_surfaceField.isInitialized())
            {
                m_surfaceField.evaluateField(samplesSurface);
            }
        }
        if (samplesVolume.samples.size() > 0)
        {
            if (m_volumeField.isInitialized())
            {
                m_volumeField.evaluateField(samplesVolume);
            }
        }
    }

    void updateFieldSurface(SampleContainer &samplesSurface) override
    {
        Timer timer;
        if (samplesSurface.samples.size() > 0)
        {
            if (!m_surfaceField.isInitialized())
            {
                m_surfaceField.buildField(samplesSurface);
            }
            else
            {
                m_surfaceField.updateField(samplesSurface);
            }
        }
        m_iteration++;
        m_timeUpdateField += timer.elapsed() * 1e-3f;
    }

    void updateFieldVolume(SampleContainer &samplesVolume) override
    {
        Timer timer;
        if (samplesVolume.samples.size() > 0)
        {
            if (!m_volumeField.isInitialized())
            {
                m_volumeField.buildField(samplesVolume);
            }
            else
            {
                m_volumeField.updateField(samplesVolume);
            }
        }
        m_iteration++;
        m_timeUpdateField += timer.elapsed() * 1e-3f;
    }

    void clearSignatures() override
    {
        m_surfaceField.clearSignatures();
        m_volumeField.clearSignatures();
    }

    void updateSubdivConfig(const PGLKDTreeArguments &cfg) override
    {
        m_surfaceField.updateSubdivConfig(cfg);
        m_volumeField.updateSubdivConfig(cfg);
    }

    void loadSubdivConfig(PGLKDTreeArguments &cfg) const override
    {
        m_surfaceField.loadSubdivConfig(cfg);
    }

    void resetField() override
    {
        m_iteration = 0;
        m_totalSPP = 0;
        m_surfaceField.resetField();
        m_volumeField.resetField();
    }

    PGL_SPATIAL_STRUCTURE_TYPE getSpatialStructureType() const override
    {
        return SurfaceFieldType::SpatialStructureBuilder::SPATIAL_STRUCTURE_TYPE;
    }

    PGL_DIRECTIONAL_DISTRIBUTION_TYPE getDirectionalDistributionType() const override
    {
        return SurfaceFieldType::DirectionalDistributionFactory::DIRECTIONAL_DISTRIBUTION_TYPE;
    }

    size_t getIteration() const override
    {
        return m_iteration;
    }

    void serialize(std::ostream &os) const override
    {
        os.write(reinterpret_cast<const char *>(&m_iteration), sizeof(m_iteration));
        os.write(reinterpret_cast<const char *>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.serialize(os);
        m_volumeField.serialize(os);
    }

    void deserialize(std::istream &is) override
    {
        is.read(reinterpret_cast<char *>(&m_iteration), sizeof(m_iteration));
        is.read(reinterpret_cast<char *>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.deserialize(is);
        m_volumeField.deserialize(is);
    }

    virtual bool validate(const bool checkSurface, const bool checkVolume) const override
    {
        bool valid = true;
        if (m_surfaceField.isInitialized())
            valid = valid & m_surfaceField.isValid();
        if (m_volumeField.isInitialized())
            valid = valid & m_volumeField.isValid();
        return valid;
    }

    void storeToFile(const std::string fieldFileName) const override
    {
        std::filebuf fb;
        fb.open(fieldFileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);

        os.write(FIELD_FILE_HEADER_STRING, strlen(FIELD_FILE_HEADER_STRING) + 1);

        auto spatialStructureType = SurfaceFieldType::SpatialStructureBuilder::SPATIAL_STRUCTURE_TYPE;
        os.write(reinterpret_cast<const char *>(&spatialStructureType), sizeof(spatialStructureType));
        auto directionalDistributionType = SurfaceFieldType::DirectionalDistributionFactory::DIRECTIONAL_DISTRIBUTION_TYPE;
        os.write(reinterpret_cast<const char *>(&directionalDistributionType), sizeof(directionalDistributionType));

        serialize(os);

        os.flush();
        fb.close();
    }

    virtual bool operator==(const ISurfaceVolumeField *b) const override
    {
        bool equal = true;
        const SurfaceVolumeField *fieldB = dynamic_cast<const SurfaceVolumeField *>(b);
        if (!fieldB || m_iteration != fieldB->m_iteration || m_totalSPP != fieldB->m_totalSPP || !m_surfaceField.operator==(fieldB->m_surfaceField) ||
            !m_volumeField.operator==(fieldB->m_volumeField))
        {
            equal = false;
        }
        return equal;
    }

    FieldStatistics *getSurfaceStatistics() const override
    {
        FieldStatistics *stats = m_surfaceField.getStatistics();
        return stats;
    }

    FieldStatistics *getVolumeStatistics() const override
    {
        FieldStatistics *stats = m_volumeField.getStatistics();
        return stats;
    }

    size_t getRegionCountSurface() const override
    {
        return m_surfaceField.getRegionCount();
    }

    size_t getLeafCountSurface() const override {
        return m_surfaceField.getLeafCount();
    }

    PGLRegionStatistics getRegionStatsSurface(uint32_t id) const override
    {
        return m_surfaceField.getRegionStats(id);
    }

    std::pair<PGLRegionStatistics, PGLRegionStatistics> getCoarseFineRegionStatsSurface(const openpgl::Point3 &pos) const override
    {
        return m_surfaceField.getCoarseFineRegionStats(pos);
    }

    std::pair<PGLDirectionalSignature, PGLDirectionalSignature> getDirectionalSignatures(const openpgl::Point3 &pos, uint32_t lookaheadDepth, uint8_t modelIndex, uint8_t &splitDim, bool &isRight) const override
    {
        return m_surfaceField.getDirectionalSignatures(pos, lookaheadDepth, modelIndex, splitDim, isRight);
    }

private:
    size_t m_iteration{0};
    size_t m_totalSPP{0};
    float m_timeUpdateField{0.0f};

    SurfaceFieldType m_surfaceField;
    VolumeFieldType m_volumeField;
};

}  // namespace openpgl
