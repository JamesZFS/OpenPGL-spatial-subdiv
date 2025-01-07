// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/Range.h"
#include "../data/SampleContainerInternal.h"
#include "../spatial/KNN.h"
#include "../spatial/Region.h"
#include "../spatial/kdtree/KDTree.h"
#include "FieldStatistics.h"

#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>
#else
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#endif
#define USE_PRECOMPUTED_NN 1

namespace openpgl
{

template <int Vecsize, class TDirectionalDistributionFactory, template <typename, typename, typename, typename> class TSpatialStructureBuilder, typename TSamplingDistribution>
struct Field {
public:
    using DirectionalDistributionFactory = TDirectionalDistributionFactory;
    using DirectionalDistributionFactorySettings = typename TDirectionalDistributionFactory::Configuration;
    using DirectionalDistribution = typename TDirectionalDistributionFactory::Distribution;

    using SampleContainer = SampleDataStorage::SampleContainer;
    // using SampleContainerInternal = ContainerInternal<SampleData>;
    // using ZeroValueSampleContainerInternal = ContainerInternal<ZeroValueSampleData>;
    using SampleContainerInternal = std::vector<SampleData>;
    using ZeroValueSampleContainerInternal = std::vector<ZeroValueSampleData>;

    typedef Region<DirectionalDistribution, typename TDirectionalDistributionFactory::Statistics> RegionType;
    typedef openpgl::Range RangeType;
    typedef std::pair<RegionType, RangeType> RegionStorageType;
    typedef tbb::concurrent_vector<RegionStorageType> RegionStorageContainerType;

    using SpatialStructureBuilder = TSpatialStructureBuilder<RegionType, SampleContainerInternal, ZeroValueSampleContainerInternal, TSamplingDistribution>;
    using SpatialStructure = typename SpatialStructureBuilder::SpatialStructure;
    using SpatialBuilderSettings = typename SpatialStructureBuilder::Settings;

    struct DebugSettings
    {
        bool fitRegions{true};
    };

    struct SpatialSettings
    {
        SpatialBuilderSettings spatialSubdivBuilderSettings;
        bool useStochasticNNLookUp{false};
        bool useISNNLookUp{false};
        bool deterministic{false};
        float decayOnSpatialSplit{0.25f};

        std::string toString() const;
    };

    struct Settings
    {
        SpatialSettings settings;
        DirectionalDistributionFactorySettings distributionFactorySettings;
        DebugSettings debugSettings;
        std::string toString() const;
    };

public:
    Field() = default;

    Field(const Field &) = delete;

    Field(const Settings &settings)
    {
        m_decayOnSpatialSplit = settings.settings.decayOnSpatialSplit;
        m_deterministic = settings.settings.deterministic;
        m_fitRegions = settings.debugSettings.fitRegions;
        m_useStochasticNNLookUp = settings.settings.useStochasticNNLookUp;
        m_useISNNLookUp = settings.settings.useISNNLookUp;
        m_spatialSubdivBuilderSettings = settings.settings.spatialSubdivBuilderSettings;

        m_distributionFactorySettings = settings.distributionFactorySettings;
    }

    ~Field()
    {
        m_regionKNNSearchTree.reset();
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds)
    {
        m_sceneBounds = sceneBounds;
        m_isSceneBoundsSet = true;
    }

    openpgl::BBox getSceneBounds() const
    {
        return m_sceneBounds;
    }

    void setIsSurface(const bool isSurface)
    {
        m_isSurface = isSurface;
    }

    inline const RegionType *getRegion(const openpgl::Point3 &p, float *sample1D, uint32_t &id) const
    {
        if (m_iteration > 0 && embree::inside(m_spatialSubdiv.getBounds(), p))
        {
            if (m_useStochasticNNLookUp && *sample1D >= 0.f)
            {
                if (USE_PRECOMPUTED_NN)
                {
                    uint32_t regionIdx = getApproximateClosestRegionIdx(m_regionKNNSearchTree, p, sample1D, id);
                    return &m_regionStorageContainer[regionIdx].first;
                }
                else
                {
                    uint32_t regionIdx = getClosestRegionIdx(m_regionKNNSearchTree, p, sample1D, id);
                    if (regionIdx != -1)
                    {
                        return &m_regionStorageContainer[regionIdx].first;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
            else
            {
                uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p);
                OPENPGL_ASSERT(dataIdx < m_regionStorageContainer.size());
                id = dataIdx;
                return &m_regionStorageContainer[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }

    uint32_t getRegionId(const openpgl::Point3 &p) const
    {
        float sample = -1;
        uint32_t id;
        auto region = getRegion(p, &sample, id);
        return region ? id : -1;
    }

    // Deprecated
    inline const RegionType *getFineRegion(const openpgl::Point3 &p, float *sample1D, uint32_t &id) const
    {
        return getRegion(p, sample1D, id);
    }

    void buildField(const SampleContainer &samples)
    {
        m_iteration = 0;
        m_totalSPP = 0;
        if (samples.samples.size() > 0)
        {
            Timer updateAll;
            Timer updateStep;

            if (samples_.capacity() < samples.samples.size())
            {
                samples_.reserve(2 * samples.samples.size());
            }
            samples_.resize(samples.samples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.samples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.samples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    samples_[i] = samples.samples[i];
            });

            if (zeroValueSamples_.capacity() < samples.zeroValueSamples.size())
            {
                zeroValueSamples_.reserve(2 * samples.zeroValueSamples.size());
            }
            zeroValueSamples_.resize(samples.zeroValueSamples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.zeroValueSamples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.zeroValueSamples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    zeroValueSamples_[i] = samples.zeroValueSamples[i];
            });
            m_timeLastUpdateCopySamples = updateStep.elapsed() * 1e-3f;

            if (!m_isSceneBoundsSet)
            {
                estimateSceneBounds(samples_);
            }
            updateStep.reset();
            buildSpatialStructure(m_sceneBounds, samples_, zeroValueSamples_);
            m_timeLastUpdateSpatialStructureUpdate = updateStep.elapsed() * 1e-3f;

            updateStep.reset();
            fitRegions(samples_, zeroValueSamples_);
            m_timeLastUpdateDirectionalDistriubtionUpdate = updateStep.elapsed() * 1e-3f;
            m_timeLastUpdate = updateAll.elapsed() * 1e-3f;
        }
        m_iteration++;
    }

    void updateField(const SampleContainer &samples)
    {
        if (samples.samples.size() > 0)
        {
            Timer updateAll;
            Timer updateStep;

            if (samples_.capacity() < samples.samples.size())
            {
                samples_.reserve(2 * samples.samples.size());
            }
            samples_.resize(samples.samples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.samples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.samples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    samples_[i] = samples.samples[i];
            });

            if (zeroValueSamples_.capacity() < samples.zeroValueSamples.size())
            {
                zeroValueSamples_.reserve(2 * samples.zeroValueSamples.size());
            }
            zeroValueSamples_.resize(samples.zeroValueSamples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.zeroValueSamples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.zeroValueSamples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    zeroValueSamples_[i] = samples.zeroValueSamples[i];
            });
            m_timeLastUpdateCopySamples = updateStep.elapsed() * 1e-3f;

            updateStep.reset();
            updateSpatialStructure(samples_, zeroValueSamples_);
            m_timeLastUpdateSpatialStructureUpdate = updateStep.elapsed() * 1e-3f;

            updateStep.reset();
            updateRegions(samples_, zeroValueSamples_);

            m_timeLastUpdateDirectionalDistriubtionUpdate = updateStep.elapsed() * 1e-3f;
            m_timeLastUpdate = updateAll.elapsed() * 1e-3f;
            std::cout << "updateField() took " << updateAll.elapsed() * 1e-3f << " ms" << std::endl;
        }
        m_iteration++;
    }

    void evaluateField(const SampleContainer &samples) {
        if (samples.samples.size() > 0)
        {
            if (samples_.capacity() < samples.samples.size())
            {
                samples_.reserve(2 * samples.samples.size());
            }
            samples_.resize(samples.samples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.samples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.samples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    samples_[i] = samples.samples[i];
            });

            if (zeroValueSamples_.capacity() < samples.zeroValueSamples.size())
            {
                zeroValueSamples_.reserve(2 * samples.zeroValueSamples.size());
            }
            zeroValueSamples_.resize(samples.zeroValueSamples.size());
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(size_t(0), samples.zeroValueSamples.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, samples.zeroValueSamples.size()), [&](tbb::blocked_range<int> r) {
#endif
                for (size_t i = r.begin(); i < r.end(); i++)
                    zeroValueSamples_[i] = samples.zeroValueSamples[i];
            });

            Timer timer;
            // Only update CE stats and signatures, no subdivision or fitting
            m_spatialSubdivBuilder.evaluateRegions(m_spatialSubdiv, samples_, m_regionStorageContainer, m_spatialSubdivBuilderSettings, *this);
            m_spatialSubdivBuilder.evaluateRegions(m_spatialSubdiv, zeroValueSamples_, m_regionStorageContainer, m_spatialSubdivBuilderSettings, *this);
            std::cout << "evaluateRegions() took " << timer.elapsed() * 1e-3f << " ms" << std::endl;
        }
    }

    void clearCEStats() {
        for (auto &region : m_regionStorageContainer) {
            region.first.ceStatistics.clear();
        }
    }

    void clearSignatures() {
        for (auto &region : m_regionStorageContainer) {
            region.first.resetSignatures();
        }
    }

    void updateSubdivConfig(const PGLKDTreeArguments &cfg)
    {
        m_useStochasticNNLookUp = cfg.knnLookup;
        m_useISNNLookUp = cfg.isKnnLookup;
        m_decayOnSpatialSplit = cfg.vmmDecay;
        m_spatialSubdivBuilderSettings.updateFromConfig(cfg);
        CEStatistics::clampValue = cfg.ceClampValue;
    }

    void loadSubdivConfig(PGLKDTreeArguments &cfg) const
    {
        cfg.knnLookup = m_useStochasticNNLookUp;
        cfg.isKnnLookup = m_useISNNLookUp;
        cfg.vmmDecay = m_decayOnSpatialSplit;
        m_spatialSubdivBuilderSettings.loadToConfig(cfg);
        cfg.ceClampValue = CEStatistics::clampValue;
    }

    void resetField()
    {
        m_iteration = 0;
        m_totalSPP = 0;

        m_isSceneBoundsSet = false;
        m_initialized = false;

        m_spatialSubdiv = SpatialStructure();
        m_regionStorageContainer.clear();
        m_regionKNNSearchTree.reset();
    }

    size_t getIteration() const
    {
        return m_iteration;
    }

    size_t getRegionCount() const
    {
        return m_regionStorageContainer.size();
    }

    size_t getLeafCount() const {
        return m_spatialSubdiv.getNumLeafs();  // return only the non-lookahead regions
    }

    PGLDirectionalSignature getDirectionalSignature(const openpgl::Point3 &pos) const {
        uint32_t id = getRegionId(pos);
        if (id < m_regionStorageContainer.size()) {
            auto &region = m_regionStorageContainer[id].first;
            if (region.hasBestCandidateSplit()) {
                auto &candidate = region.getBestCandidateSplit();
                return PGLDirectionalSignature(candidate.signaturesLR[pos[region.bestSplitDim] >= candidate.pos]);
            }
        }
        return {};
    }

    uint8_t getDirectionalSignatureBestDim(const openpgl::Point3 &pos) const {
        uint32_t id = getRegionId(pos);
        if (id < m_regionStorageContainer.size()) {
            auto &region = m_regionStorageContainer[id].first;
            return region.bestSplitDim;
        }
        return 3;  // invalid
    }

    std::pair<PGLDirectionalSignature, PGLDirectionalSignature> getLRDirectionalSignatures(const openpgl::Point3 &pos, uint8_t dim) const {
        // dim == 3 => getBestDim
        uint32_t id = getRegionId(pos);
        if (id < m_regionStorageContainer.size()) {
            auto &region = m_regionStorageContainer[id].first;
            if (dim < 3) {
                auto &candidate = region.candidateSplits[dim];
                if (candidate.valid())
                    return {PGLDirectionalSignature(candidate.signaturesLR[0]), PGLDirectionalSignature(candidate.signaturesLR[1])};
            } else {
                if (region.hasBestCandidateSplit()) {
                    auto &candidate = region.getBestCandidateSplit();
                    return {PGLDirectionalSignature(candidate.signaturesLR[0]), PGLDirectionalSignature(candidate.signaturesLR[1])};
                }
            }
        }
        return {};
    }

    PGLRegionStatistics getRegionStats(uint32_t id) const
    {
        PGLRegionStatistics stats{.id = id, .fluence = 0, .crossEntropy = std::numeric_limits<float>::quiet_NaN(), .energy = std::numeric_limits<float>::quiet_NaN()};
        if (id >= m_regionStorageContainer.size())
            return stats;
        auto &region = m_regionStorageContainer[id].first;
        stats.removed = false;
        // stats.numSamples = region.ceStatistics.getNumSamples();
        stats.numSamples = region.sampleStatistics.numSamples;
        stats.numZeroValueSamples = region.sampleStatistics.numZeroValueSamples;
        stats.depth = region.depth;
        if (stats.numSamples > 0) {
            stats.fluence = region.ceStatistics.getFluence();
            stats.crossEntropy = region.ceStatistics.getCE();
        }
        stats.hasCandidateSplit = region.hasBestCandidateSplit();
        if (stats.hasCandidateSplit) {
            auto &candidate = region.getBestCandidateSplit();
            stats.energy = candidate.energy;
            stats.splitDim = region.bestSplitDim;
            stats.splitPos = candidate.pos;
        }
        // stats.sampleMean = {region.sampleStatistics.getMean().x, region.sampleStatistics.getMean().y, region.sampleStatistics.getMean().z};
        // stats.sampleVariance = {region.sampleStatistics.getVariance().x, region.sampleStatistics.getVariance().y, region.sampleStatistics.getVariance().z};
        stats.lowerBounds = {region.regionBounds.lower.x, region.regionBounds.lower.y, region.regionBounds.lower.z};
        stats.upperBounds = {region.regionBounds.upper.x, region.regionBounds.upper.y, region.regionBounds.upper.z};
        return stats;
    }

    // ! Deprecated API
    std::pair<PGLRegionStatistics, PGLRegionStatistics> getCoarseFineRegionStats(const openpgl::Point3 &pos) const
    {
        PGLRegionStatistics cStats{.id = (uint32_t) -1, .fluence = 0, .crossEntropy = std::numeric_limits<float>::quiet_NaN()};
        PGLRegionStatistics fStats = cStats;
        uint32_t cId = getRegionId(pos);
        if (cId >= m_regionStorageContainer.size())
            return {cStats, fStats};
        cStats = getRegionStats(cId);
        auto &region = m_regionStorageContainer[cId].first;
        if (cStats.hasCandidateSplit) {
            fStats.depth = region.depth + 1;
            fStats.lowerBounds = cStats.lowerBounds, fStats.upperBounds = cStats.upperBounds;
            auto &candidate = region.getBestCandidateSplit();
            if (pos[region.bestSplitDim] >= candidate.pos) {
                fStats.id = cId | (1 << 31);  // a fake distinct ID
                fStats.numSamples = candidate.signaturesLR[1].getNumSamples();
                fStats.lowerBounds[region.bestSplitDim] = candidate.pos;
            } else {
                fStats.id = cId | (1 << 30);
                fStats.numSamples = candidate.signaturesLR[0].getNumSamples();
                fStats.upperBounds[region.bestSplitDim] = candidate.pos;
            }
        }
        return {cStats, fStats};
    }

    void serialize(std::ostream &os) const
    {
        os.write(reinterpret_cast<const char *>(&m_isSurface), sizeof(m_isSurface));
        os.write(reinterpret_cast<const char *>(&m_decayOnSpatialSplit), sizeof(m_decayOnSpatialSplit));
        os.write(reinterpret_cast<const char *>(&m_iteration), sizeof(m_iteration));
        os.write(reinterpret_cast<const char *>(&m_totalSPP), sizeof(m_totalSPP));
        os.write(reinterpret_cast<const char *>(&m_deterministic), sizeof(m_deterministic));
        os.write(reinterpret_cast<const char *>(&m_fitRegions), sizeof(m_fitRegions));
        os.write(reinterpret_cast<const char *>(&m_isSceneBoundsSet), sizeof(m_isSceneBoundsSet));
        os.write(reinterpret_cast<const char *>(&m_sceneBounds), sizeof(m_sceneBounds));
        os.write(reinterpret_cast<const char *>(&m_initialized), sizeof(m_initialized));

        os.write(reinterpret_cast<const char *>(&m_timeLastUpdate), sizeof(m_timeLastUpdate));
        os.write(reinterpret_cast<const char *>(&m_timeLastUpdateCopySamples), sizeof(m_timeLastUpdateCopySamples));
        os.write(reinterpret_cast<const char *>(&m_timeLastUpdateSpatialStructureUpdate), sizeof(m_timeLastUpdateSpatialStructureUpdate));
        os.write(reinterpret_cast<const char *>(&m_timeLastUpdateDirectionalDistriubtionUpdate), sizeof(m_timeLastUpdateDirectionalDistriubtionUpdate));

        m_distributionFactorySettings.serialize(os);
        m_spatialSubdivBuilderSettings.serialize(os);
        m_spatialSubdiv.serialize(os);
        size_t size = m_regionStorageContainer.size();
        os.write(reinterpret_cast<const char *>(&size), sizeof(size));
        for (size_t i = 0; i < size; i++)
        {
            m_regionStorageContainer[i].first.serialize(os);
            m_regionStorageContainer[i].second.serialize(os);
        }
        os.write(reinterpret_cast<const char *>(&m_useStochasticNNLookUp), sizeof(m_useStochasticNNLookUp));
        os.write(reinterpret_cast<const char *>(&m_useISNNLookUp), sizeof(m_useISNNLookUp));
        m_regionKNNSearchTree.serialize(os);
        os.write(reinterpret_cast<const char *>(&g_opgl_octahedral_resolution), sizeof(g_opgl_octahedral_resolution));
        os.write(reinterpret_cast<const char *>(&g_opgl_signature_size), sizeof(g_opgl_signature_size));
    }

    void deserialize(std::istream &is)
    {
        is.read(reinterpret_cast<char *>(&m_isSurface), sizeof(m_isSurface));
        is.read(reinterpret_cast<char *>(&m_decayOnSpatialSplit), sizeof(m_decayOnSpatialSplit));
        is.read(reinterpret_cast<char *>(&m_iteration), sizeof(m_iteration));
        is.read(reinterpret_cast<char *>(&m_totalSPP), sizeof(m_totalSPP));
        is.read(reinterpret_cast<char *>(&m_deterministic), sizeof(m_deterministic));
        is.read(reinterpret_cast<char *>(&m_fitRegions), sizeof(m_fitRegions));
        is.read(reinterpret_cast<char *>(&m_isSceneBoundsSet), sizeof(m_isSceneBoundsSet));
        is.read(reinterpret_cast<char *>(&m_sceneBounds), sizeof(m_sceneBounds));
        is.read(reinterpret_cast<char *>(&m_initialized), sizeof(m_initialized));

        is.read(reinterpret_cast<char *>(&m_timeLastUpdate), sizeof(m_timeLastUpdate));
        is.read(reinterpret_cast<char *>(&m_timeLastUpdateCopySamples), sizeof(m_timeLastUpdateCopySamples));
        is.read(reinterpret_cast<char *>(&m_timeLastUpdateSpatialStructureUpdate), sizeof(m_timeLastUpdateSpatialStructureUpdate));
        is.read(reinterpret_cast<char *>(&m_timeLastUpdateDirectionalDistriubtionUpdate), sizeof(m_timeLastUpdateDirectionalDistriubtionUpdate));

        m_distributionFactorySettings.deserialize(is);
        m_spatialSubdivBuilderSettings.deserialize(is);
        m_spatialSubdiv.deserialize(is);
        size_t size;
        is.read(reinterpret_cast<char *>(&size), sizeof(size));
        m_regionStorageContainer.clear();
        m_regionStorageContainer.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            m_regionStorageContainer.emplace_back();
            m_regionStorageContainer[i].first.deserialize(is);
            m_regionStorageContainer[i].second.deserialize(is);
        }
        is.read(reinterpret_cast<char *>(&m_useStochasticNNLookUp), sizeof(m_useStochasticNNLookUp));
        is.read(reinterpret_cast<char *>(&m_useISNNLookUp), sizeof(m_useISNNLookUp));
        m_regionKNNSearchTree.deserialize(is);

        if (m_useStochasticNNLookUp && USE_PRECOMPUTED_NN && m_regionKNNSearchTree.isBuild())
        {
            m_regionKNNSearchTree.buildRegionNeighbours();
        }
        is.read(reinterpret_cast<char *>(&g_opgl_octahedral_resolution), sizeof(g_opgl_octahedral_resolution));
        is.read(reinterpret_cast<char *>(&g_opgl_signature_size), sizeof(g_opgl_signature_size));
    }

    bool isValid() const
    {
        bool valid = true;
        size_t nGuidingRegions = m_regionStorageContainer.size();
        for (int n = 0; n < nGuidingRegions; n++)
        {
            valid = valid && m_regionStorageContainer[n].first.isValid() && m_regionStorageContainer[n].first.valid;
            valid = valid && m_regionStorageContainer[n].second.isValid();
        }
        return valid;
    }

    bool isInitialized() const
    {
        return m_initialized;
    }

   private:
    void estimateSceneBounds(const SampleContainerInternal &samples)
    {
        m_sceneBounds.lower = Vector3(std::numeric_limits<float>::max());
        m_sceneBounds.upper = Vector3(std::numeric_limits<float>::min());
        m_isSceneBoundsSet = false;

        if (samples.size() > 0)
        {
            // TODO parallize this part (also use some stats?)
            for (const auto &sample : samples)
            {
                m_sceneBounds.extend(Vector3(sample.position.x, sample.position.y, sample.position.z));
            }
            Vector3 center = m_sceneBounds.center();
            m_sceneBounds.lower = center + 3.0f * (m_sceneBounds.lower - center);
            m_sceneBounds.upper = center + 3.0f * (m_sceneBounds.upper - center);
            m_isSceneBoundsSet = true;
        }
    }

    inline uint32_t getClosestRegionIdx(const KNearestRegionsSearchTree<Vecsize> &knnTree, const openpgl::Point3 &p, float *sample, uint32_t &id) const
    {
        OPENPGL_ASSERT(knnTree.isBuild());
        const uint32_t regionIdx = knnTree.sampleClosestRegionIdx(p, sample);
        return regionIdx;
    }

    inline uint32_t getApproximateClosestRegionIdx(const KNearestRegionsSearchTree<Vecsize> &knnTree, const openpgl::Point3 &p, float *sample, uint32_t &id) const
    {
        OPENPGL_ASSERT(knnTree.isBuildNeighbours());
        uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p);
        OPENPGL_ASSERT(dataIdx < m_regionStorageContainer.size());
        id = dataIdx;
        if (m_useISNNLookUp)
            return knnTree.sampleApproximateClosestRegionIdxIS(dataIdx, p, sample);
        else
            return knnTree.sampleApproximateClosestRegionIdx(dataIdx, p, sample);
    }

    inline void buildSpatialStructure(const BBox &bounds, SampleContainerInternal &samples, ZeroValueSampleContainerInternal &zeroValueSamples)
    {
        m_spatialSubdivBuilder.build(m_spatialSubdiv, bounds, samples, zeroValueSamples, m_regionStorageContainer, m_spatialSubdivBuilderSettings);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.buildRegionSearchTree(m_regionStorageContainer);
            if (USE_PRECOMPUTED_NN)
            {
                m_regionKNNSearchTree.buildRegionNeighbours();
            }
        }
    }

    inline void updateSpatialStructure(SampleContainerInternal &samples, ZeroValueSampleContainerInternal &zeroValueSamples)
    {
        Timer timer;
        m_spatialSubdivBuilder.update(m_spatialSubdiv, samples, zeroValueSamples, m_regionStorageContainer, m_spatialSubdivBuilderSettings);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.reset();
            m_regionKNNSearchTree.buildRegionSearchTree(m_regionStorageContainer);
            if (USE_PRECOMPUTED_NN)
            {
                m_regionKNNSearchTree.buildRegionNeighbours();
            }
        }
        std::cout << "updateTree() took " << timer.elapsed() * 1e-3f << " ms" << std::endl;
    }

    inline void fitRegions(SampleContainerInternal &samples, ZeroValueSampleContainerInternal &zeroValueSamples)
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
#if defined(OPENPGL_SHOW_PRINT_OUTS)
        std::cout << "fitRegion: " << (m_isSurface ? "surface" : "volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#endif

#ifdef USE_EMBREE_PARALLEL
        embree::parallel_for(0, (int)nGuidingRegions, 1, [&](const embree::range<unsigned> &r) {
            for (size_t n = r.begin(); n < r.end(); n++)
#else
        tbb::parallel_for(tbb::blocked_range<int>(0, nGuidingRegions), [&](tbb::blocked_range<int> r) {
            for (int n = r.begin(); n < r.end(); ++n)
#endif
            {
                RegionStorageType &regionStorage = m_regionStorageContainer[n];
                // TODO: replace with the region pivot for consistency
                openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.getMean();
                if (regionStorage.second.size() > 0)
                {
                    if (m_deterministic)
                    {
                        std::sort(samples.begin() + regionStorage.second.m_begin, samples.begin() + regionStorage.second.m_end, SampleDataLess);
                    }

                    if (m_fitRegions)
                    {
                        typename DirectionalDistributionFactory::FittingStatistics fittingStats;
                        m_distributionFactory.prepareSamples(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin,
                                                             regionStorage.first.sampleStatistics, m_distributionFactorySettings);
                        m_distributionFactory.fit(regionStorage.first.distribution, regionStorage.first.trainingStatistics, samples.data() + regionStorage.second.m_begin,
                                                  regionStorage.second.m_end - regionStorage.second.m_begin, m_distributionFactorySettings, fittingStats);
#ifdef OPENPGL_RADIANCE_CACHES
                        m_distributionFactory.updateFluenceEstimate(regionStorage.first.distribution, samples.data() + regionStorage.second.m_begin,
                                                                    regionStorage.second.m_end - regionStorage.second.m_begin, regionStorage.first.numZeroValueSamples,
                                                                    regionStorage.first.sampleStatistics);
                        regionStorage.first.outRadianceHist.update(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin,
                                                                   zeroValueSamples.data() + regionStorage.second.m_is_begin,
                                                                   regionStorage.second.m_is_end - regionStorage.second.m_is_begin);
#endif
                        // TODO: we should move setting the pivot to the factory
                        regionStorage.first.distribution._pivotPosition = sampleMean;
                        regionStorage.first.valid = regionStorage.first.distribution.isValid();
#ifdef OPENPGL_DEBUG_MODE
                        if (!regionStorage.first.valid)
                            std::cout << "!!!! " << (m_isSurface ? "Surface" : "Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString()
                                      << std::endl;
#endif
                        regionStorage.first.splitFlag = false;
                        regionStorage.first.initialized = true;
                    }
                }
                else
                {
                    regionStorage.first.valid = true;
                    regionStorage.first.initialized = false;
                    regionStorage.first.splitFlag = false;
                }
                regionStorage.second.reset();
                OPENPGL_ASSERT(regionStorage.first.isValid());
            }
        });
        m_initialized = true;
        OPENPGL_ASSERT(this->isValid());
    }

    void updateRegions(SampleContainerInternal &samples, ZeroValueSampleContainerInternal &zeroValueSamples)
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
#if defined(OPENPGL_SHOW_PRINT_OUTS)
        std::cout << "updateRegion: " << (m_isSurface ? "surface" : "volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#endif
#ifdef USE_EMBREE_PARALLEL
        embree::parallel_for(0, (int)nGuidingRegions, 1, [&](const embree::range<unsigned> &r) {
            for (size_t n = r.begin(); n < r.end(); n++)
#else
        tbb::parallel_for(tbb::blocked_range<int>(0, nGuidingRegions), [&](tbb::blocked_range<int> r) {
            for (int n = r.begin(); n < r.end(); ++n)
#endif
            {
                RegionStorageType &regionStorage = m_regionStorageContainer[n];
                if (regionStorage.first.splitFlag)
                {
                    regionStorage.first.distribution.decay(this->m_decayOnSpatialSplit);
                    regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
#ifdef OPENPGL_RADIANCE_CACHES
                    regionStorage.first.outRadianceHist.decay(this->m_decayOnSpatialSplit);
#endif
                    regionStorage.first.splitFlag = false;
                }

                // TODO: replace with the region pivot for consistency
                openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.getMean();
                if (regionStorage.second.size() > 0)
                {
#ifdef OPENPGL_DEBUG_MODE
                    RegionType oldRegion = regionStorage.first;
#endif
                    if (m_deterministic)
                    {
                        std::sort(samples.begin() + regionStorage.second.m_begin, samples.begin() + regionStorage.second.m_end, SampleDataLess);
                    }

                    if (m_fitRegions)
                    {
                        // TODO: we should move applying the paralax comp to the Distribution to the factory
                        if (regionStorage.first.initialized && DirectionalDistribution::ParallaxCompensation == 1)
                        {
                            regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution,
                                                                                                           regionStorage.first.distribution._pivotPosition - sampleMean);
                            regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
                            OPENPGL_ASSERT(regionStorage.first.distribution.isValid());
                            OPENPGL_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
                        }
                        typename DirectionalDistributionFactory::FittingStatistics fittingStats;
                        m_distributionFactory.prepareSamples(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin,
                                                             regionStorage.first.sampleStatistics, m_distributionFactorySettings);
                        if (regionStorage.first.initialized)
                        {
                            m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, samples.data() + regionStorage.second.m_begin,
                                                     regionStorage.second.m_end - regionStorage.second.m_begin, m_distributionFactorySettings, fittingStats);
                        }
                        else
                        {
                            m_distributionFactory.fit(regionStorage.first.distribution, regionStorage.first.trainingStatistics, samples.data() + regionStorage.second.m_begin,
                                                     regionStorage.second.m_end - regionStorage.second.m_begin, m_distributionFactorySettings, fittingStats);
                            regionStorage.first.initialized = true;
                        }
#ifdef OPENPGL_RADIANCE_CACHES
                        m_distributionFactory.updateFluenceEstimate(regionStorage.first.distribution, samples.data() + regionStorage.second.m_begin,
                                                                    regionStorage.second.m_end - regionStorage.second.m_begin, regionStorage.first.numZeroValueSamples,
                                                                    regionStorage.first.sampleStatistics);
                        regionStorage.first.outRadianceHist.update(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin,
                                                                   zeroValueSamples.data() + regionStorage.second.m_is_begin,
                                                                   regionStorage.second.m_is_end - regionStorage.second.m_is_begin);
#endif
                        regionStorage.first.valid = regionStorage.first.isValid();
#ifdef OPENPGL_DEBUG_MODE
                        if (!regionStorage.first.valid)
                        {
                            std::cout << "!!!! " << (m_isSurface ? "Surface" : "Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString()
                                      << std::endl;
                            storeInvalidRegionData("regionBeforeUpdate_" + std::string((m_isSurface ? "surf" : "vol")) + "_itr" + std::to_string(this->m_iteration) + "_region" +
                                                       std::to_string(n) + ".dump",
                                                   oldRegion, dataPoints, m_distributionFactorySettings);
                        }
#endif
                    }
                }
                else
                {
                    RegionStorageType &regionStorage = m_regionStorageContainer[n];
                    if (regionStorage.first.splitFlag)
                    {
                        regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
                        regionStorage.first.splitFlag = false;
                    }
                }
                regionStorage.second.reset();
                OPENPGL_ASSERT(regionStorage.first.isValid());
            }
        });
        OPENPGL_ASSERT(this->isValid());
    }

    static void storeInvalidRegionData(const std::string &fileName, const RegionType &regionBeforeUpdate, const std::vector<SampleData> &samples,
                                       const DirectionalDistributionFactorySettings &factorySettings)
    {
        std::filebuf fbDump;
        fbDump.open(fileName, std::ios::out);
        std::ostream dumpStream(&fbDump);

        factorySettings.serialize(dumpStream);

        size_t numSamples = samples.size();
        dumpStream.write(reinterpret_cast<const char *>(&numSamples), sizeof(size_t));
        for (size_t i = 0; i < numSamples; i++)
        {
            dumpStream.write(reinterpret_cast<const char *>(&samples[i]), sizeof(openpgl::SampleData));
        }

        regionBeforeUpdate.serialize(dumpStream);
        fbDump.close();
    }

   public:
    static void loadInvalidRegionData(const std::string &fileName, RegionType &regionBeforeUpdate, std::vector<SampleData> &samples,
                                      DirectionalDistributionFactorySettings &factorySettings)
    {
        std::filebuf fbDumpIn;
        fbDumpIn.open(fileName, std::ios::in);
        std::istream dumpIStream(&fbDumpIn);

        factorySettings.deserialize(dumpIStream);
        samples.clear();
        size_t numSamples;
        dumpIStream.read(reinterpret_cast<char *>(&numSamples), sizeof(size_t));
        for (size_t i = 0; i < numSamples; i++)
        {
            SampleData dsd;
            dumpIStream.read(reinterpret_cast<char *>(&dsd), sizeof(openpgl::SampleData));
            samples.push_back(dsd);
        }
        regionBeforeUpdate.deserialize(dumpIStream);
        fbDumpIn.close();
    }

    bool operator==(const Field &b) const
    {
        bool equal = true;
        if (!m_initialized && !b.m_initialized)
            return true;
        if (m_isSurface != b.m_isSurface || m_decayOnSpatialSplit != b.m_decayOnSpatialSplit || m_iteration != b.m_iteration || m_totalSPP != b.m_totalSPP ||
            /*m_numThreads != b.m_numThreads ||*/ m_deterministic != b.m_deterministic || m_fitRegions != b.m_fitRegions || m_isSceneBoundsSet != b.m_isSceneBoundsSet ||
            m_sceneBounds.lower.x != b.m_sceneBounds.lower.x || m_sceneBounds.lower.y != b.m_sceneBounds.lower.y || m_sceneBounds.lower.z != b.m_sceneBounds.lower.z ||
            m_sceneBounds.upper.x != b.m_sceneBounds.upper.x || m_sceneBounds.upper.y != b.m_sceneBounds.upper.y || m_sceneBounds.upper.z != b.m_sceneBounds.upper.z ||
            m_initialized != b.m_initialized || !m_distributionFactorySettings.operator==(b.m_distributionFactorySettings) ||
            !m_spatialSubdivBuilderSettings.operator==(b.m_spatialSubdivBuilderSettings) ||
            //! m_spatialSubdiv.operator==(b.m_spatialSubdiv) ||
            m_useStochasticNNLookUp != b.m_useStochasticNNLookUp)
        {
            equal = false;
        }

        std::vector<uint32_t> dataStorageIndicesA;
        std::vector<KDNode> treeNodesA;
        m_spatialSubdiv.rearrangeNodesForCompare(treeNodesA, dataStorageIndicesA);

        std::vector<uint32_t> dataStorageIndicesB;
        std::vector<KDNode> treeNodesB;
        b.m_spatialSubdiv.rearrangeNodesForCompare(treeNodesB, dataStorageIndicesB);

        if (dataStorageIndicesA.size() != dataStorageIndicesB.size() || treeNodesA.size() != treeNodesB.size() ||
            m_regionStorageContainer.size() != b.m_regionStorageContainer.size())
        {
            equal = false;
            return equal;
        }

        for (int n = 0; n < treeNodesA.size(); n++)
        {
            if (!treeNodesA[n].operator==(treeNodesB[n]))
            {
                equal = false;
            }
        }

        for (int n = 0; n < dataStorageIndicesA.size(); n++)
        {
            size_t idxA = dataStorageIndicesA[n];
            size_t idxB = dataStorageIndicesB[n];
            if (!m_regionStorageContainer[idxA].second.operator==(b.m_regionStorageContainer[idxB].second) ||
                !m_regionStorageContainer[idxA].first.operator==(b.m_regionStorageContainer[idxB].first))
            {
                equal = false;
            }
        }
        return equal;
    }

    FieldStatistics *getStatistics() const
    {
        FieldStatistics *stats = new FieldStatistics();
        stats->numCacheRegions = m_regionStorageContainer.size();
        stats->numCacheRegionsReserved = m_regionStorageContainer.capacity();
        stats->sizePerCacheRegions = sizeof(RegionStorageType);
        stats->sizeAllCacheRegionsUsed = sizeof(RegionStorageType) * m_regionStorageContainer.size();
        stats->sizeAllCacheRegionsReserved = sizeof(RegionStorageType) * m_regionStorageContainer.capacity();
        stats->timeLastUpdate = m_timeLastUpdate;
        stats->timeLastUpdateCopySamples = m_timeLastUpdateCopySamples;
        stats->timeLastUpdateSpatialStructureUpdate = m_timeLastUpdateSpatialStructureUpdate;
        stats->timeLastUpdateDirectionalDistriubtionUpdate = m_timeLastUpdateDirectionalDistriubtionUpdate;

        stats->spatialStructureStatistics = m_spatialSubdiv.getStatistics();

        stats->directionalDistributionStatistics.sizePerDistribution = sizeof(DirectionalDistribution);
        stats->directionalDistributionStatistics.minNumberOfComponents = 1e10f;
        stats->directionalDistributionStatistics.maxNumberOfComponents = 0.0f;
        stats->directionalDistributionStatistics.averageNumberOfComponents = 0.0f;
        stats->directionalDistributionStatistics.secondMomentNumberOfComponents = 0.0f;

        int numDistributions = m_regionStorageContainer.size();
        for (int i = 0; i < numDistributions; i++)
        {
            int numDistributionComponents = m_regionStorageContainer[i].first.distribution.getNumComponents();
            stats->directionalDistributionStatistics.minNumberOfComponents =
                std::min(stats->directionalDistributionStatistics.minNumberOfComponents, (float)numDistributionComponents);
            stats->directionalDistributionStatistics.maxNumberOfComponents =
                std::max(stats->directionalDistributionStatistics.maxNumberOfComponents, (float)numDistributionComponents);
            stats->directionalDistributionStatistics.averageNumberOfComponents += numDistributionComponents;
            stats->directionalDistributionStatistics.secondMomentNumberOfComponents += numDistributionComponents * numDistributionComponents;
        }
        stats->directionalDistributionStatistics.averageNumberOfComponents /= float(numDistributions);
        stats->directionalDistributionStatistics.secondMomentNumberOfComponents /= float(numDistributions);
        stats->directionalDistributionStatistics.secondMomentNumberOfComponents = std::sqrt(stats->directionalDistributionStatistics.secondMomentNumberOfComponents);
        return stats;
    }

   private:
    bool m_isSurface{true};

    float m_decayOnSpatialSplit{0.25f};

    size_t m_iteration{0};
    size_t m_totalSPP{0};

    // flag to deactivate the training of the directional distributions (i.e., for benchmarking the spatial structure build)
    bool m_fitRegions{true};
    // if the fitting process should be deterministic (i.e, samples are sorted before training)
    bool m_deterministic{false};

    bool m_isSceneBoundsSet{false};
    BBox m_sceneBounds;

    bool m_initialized{false};

    DirectionalDistributionFactory m_distributionFactory;
    DirectionalDistributionFactorySettings m_distributionFactorySettings;
    /////////////////////////////////////////////////////////
    ///////// Spatial Structure        //////////////////////
    /////////////////////////////////////////////////////////

    SpatialStructureBuilder m_spatialSubdivBuilder;
    SpatialBuilderSettings m_spatialSubdivBuilderSettings;

    SpatialStructure m_spatialSubdiv;
    RegionStorageContainerType m_regionStorageContainer;

    bool m_useStochasticNNLookUp{false};
    bool m_useISNNLookUp{false};
    KNearestRegionsSearchTree<Vecsize> m_regionKNNSearchTree;

    SampleContainerInternal samples_;
    ZeroValueSampleContainerInternal zeroValueSamples_;

    float m_timeLastUpdate{0.f};
    float m_timeLastUpdateCopySamples{0.f};
    float m_timeLastUpdateSpatialStructureUpdate{0.f};
    float m_timeLastUpdateDirectionalDistriubtionUpdate{0.f};
};

}  // namespace openpgl
