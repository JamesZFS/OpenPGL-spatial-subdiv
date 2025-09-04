// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "KDTree.h"
#include "../../data/SampleStatistics.h"
#include "../../data/Range.h"
#include "../../include/openpgl/types.h"

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_invoke.h>
#include <embreeSrc/common/algorithms/parallel_reduce.h>
#include <embreeSrc/common/algorithms/parallel_partition.h>

#include <iostream>
#include <limits>
#include <random>

// #define THRESHOLD_VAR_RATIO         1e-3
#define PGL_SIGNATURE_MAX_SAMPLES   8e6


namespace openpgl
{


template <typename T, typename = std::void_t<>>
struct has_member_weight : std::false_type {};

template <typename T>
struct has_member_weight<T, std::void_t<decltype(std::declval<T>().weight)>> : std::true_type {};

template <typename T, typename = std::void_t<>>
struct has_function_applyCosineProduct : std::false_type {};

template <typename T>
struct has_function_applyCosineProduct<T, std::void_t<decltype(std::declval<T>().applyCosineProduct(std::declval<Vector3>()))>> : std::true_type {};

template <typename F, typename G>
inline void invoke(const F &f, const G &g) {
#if NDEBUG
    tbb::parallel_invoke(f, g);
#else
    f();
    g();
#endif
}

template <typename TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
struct KDTreePartitionBuilder
{
    const static PGL_SPATIAL_STRUCTURE_TYPE SPATIAL_STRUCTURE_TYPE = PGL_SPATIAL_STRUCTURE_KDTREE;

    typedef KDTree SpatialStructure;
    using RegionType = TRegion;

#ifdef USE_EMBREE_PARALLEL
    static const size_t PARALLEL_THRESHOLD = 4 * 1024;
    static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 4 * 1024;
#endif

    struct Settings
    {
        uint32_t maxDepth {32};
        uint32_t minSamplesCandidateSplit {1000};  // to ensure the proposed split position is good enough
        uint32_t minSamplesPromotion {1000};  // to ensure the variance of signature estimates are small enough
        uint32_t sampleCountThreshold {PGL_TREE_MAX_SAMPLE_PER_LEAF};  // threshold of OpenPGL's standard subdivision scheme
        uint32_t initializingIters {1};  // the number of iterations to use the standard subdivision scheme, after which the signature threshold kicks in
        uint32_t lookaheadDepth {6};  // levels of lookahead
        float signatureDistanceThreshold {0.15f};  // triggers promotion if the distance between the signatures of the left and right children is greater than this threshold
        bool enablePromotion {true};
        float sufficientCriterionThreshold {1e-4f};  // Phi^{-1}(1 - alpha)
        float angularDistanceThreshold {3.f * M_PIf / 180.f};  // 3 degrees by default
        float angularAlpha {1e-4};  // the 100(1-alpha)% confidence interval is used
        float knnJitterMultiplier {1.0f};
        bool reproject {false};  // whether to reproject samples to the center of the parent region when calculating signatures
        PGL_SPATIAL_ANGULAR_TYPE angularType {PGL_SPATIAL_ANGULAR_SERIES};  // angular criterion type
        PGL_SPATIAL_KNN_TYPE knnType {PGL_SPATIAL_KNN_UNIFORM};  // stochastic query strategy

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;

        bool operator==(const Settings &b) const
        {
            return maxDepth == b.maxDepth && minSamplesCandidateSplit == b.minSamplesCandidateSplit &&
                   minSamplesPromotion == b.minSamplesPromotion && sampleCountThreshold == b.sampleCountThreshold &&
                   initializingIters == b.initializingIters && lookaheadDepth == b.lookaheadDepth &&
                   signatureDistanceThreshold == b.signatureDistanceThreshold &&
                   enablePromotion == b.enablePromotion &&
                   sufficientCriterionThreshold == b.sufficientCriterionThreshold && angularDistanceThreshold == b.angularDistanceThreshold &&
                   angularAlpha == b.angularAlpha && knnJitterMultiplier == b.knnJitterMultiplier && reproject == b.reproject && angularType == b.angularType && knnType == b.knnType;
        }

        void updateFromConfig(const PGLKDTreeArguments &cfg)
        {
            maxDepth = cfg.maxDepth;
            minSamplesCandidateSplit = cfg.minSamplesCandidateSplit;
            minSamplesPromotion = cfg.minSamplesPromotion;
            sampleCountThreshold = cfg.sampleCountThreshold;
            initializingIters = cfg.initializingIters;
            lookaheadDepth = cfg.lookaheadDepth;
            signatureDistanceThreshold = cfg.signatureDistanceThreshold;
            sufficientCriterionThreshold = PhiInv(1.f - cfg.fluenceAlpha);
            angularDistanceThreshold = cfg.angularDistanceThreshold;
            angularAlpha = cfg.angularAlpha;
            knnJitterMultiplier = cfg.knnJitterMultiplier;
            reproject = cfg.reproject;
            enablePromotion = cfg.enablePromotion;
            angularType = cfg.angularType;
            knnType = cfg.knnType;
        }

        void loadToConfig(PGLKDTreeArguments &cfg) const
        {
            cfg.maxDepth = maxDepth;
            cfg.minSamplesCandidateSplit = minSamplesCandidateSplit;
            cfg.minSamplesPromotion = minSamplesPromotion;
            cfg.sampleCountThreshold = sampleCountThreshold;
            cfg.initializingIters = initializingIters;
            cfg.lookaheadDepth = lookaheadDepth;
            cfg.signatureDistanceThreshold = signatureDistanceThreshold;
            cfg.fluenceAlpha = 1.f - Phi(sufficientCriterionThreshold);
            cfg.angularDistanceThreshold = angularDistanceThreshold;
            cfg.angularAlpha = angularAlpha;
            cfg.reproject = reproject;
            cfg.knnJitterMultiplier = knnJitterMultiplier;
            cfg.enablePromotion = enablePromotion;
            cfg.angularType = angularType;
            cfg.knnType = knnType;
        }
    };

    void build(KDTree &kdTree, const BBox &bounds, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples,
        tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, CandidateRegionStorage &candidateDataStorage,
        const Settings &buildSettings, uint32_t iteration) const
    {
        std::cout << buildSettings.toString() << std::endl;

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;
        dataStorage[0].first.candidate.signature.clear();

        update(kdTree, samples, zeroSamples, dataStorage, candidateDataStorage, buildSettings, iteration);
    }

    void update(KDTree &kdTree, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples,
        tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, CandidateRegionStorage &candidateDataStorage,
        const Settings &buildSettings, uint32_t iteration) const
    {
        Timer timer;
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.sampleCountThreshold+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        KDNode &root = kdTree.getRoot();
        BBox bounds;
        bounds = kdTree.getBounds();
        std::cout << "Total bounds " << bounds << std::endl;

        updateTreeNode(kdTree, root, 1, bounds,
            samples, Range(0, samples.size()), zeroSamples, Range(0, zeroSamples.size()),
            dataStorage, candidateDataStorage,
            buildSettings, iteration);

        if (iteration >= buildSettings.initializingIters) {
            // Postprocessing: clear flags, decay signature when a region has way too many samples
            embree::parallel_for(dataStorage.size(), [&](embree::range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    dataStorage[i].first.candidate.updated = false;
                    if (dataStorage[i].first.candidate.signature.numSamples > PGL_SIGNATURE_MAX_SAMPLES)
                        dataStorage[i].first.candidate.signature.decay(0.5f);
                }
            });
            embree::parallel_for(candidateDataStorage.size(), [&](embree::range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    candidateDataStorage[i].updated = false;
                    if (candidateDataStorage[i].signature.numSamples > PGL_SIGNATURE_MAX_SAMPLES)
                        candidateDataStorage[i].signature.decay(0.5f);
                }
            });
        }
        kdTree.finalize();
        double updateElapsed = timer.elapsed();
        std::cout << "KDTreePartitionBuilder::update() total update took " << updateElapsed * 1e-6 << " s, "
            << "total valid regions: " << kdTree.getNumLeafs() << std::endl;
    }

    template<class TContainer, class FieldType>
    void evaluateRegions(KDTree &kdTree, TContainer &samples,
        tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, CandidateRegionStorage &candidateDataStorage,
        const Settings &buildSettings, const FieldType &field) {

        KDNode &root = kdTree.getRoot();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        evaluateRegionsNode(kdTree, root, 1, samples, sampleRange,
            dataStorage, candidateDataStorage,
            buildSettings, field);

        // Postprocessing: decay signature when a region has way too many samples
        embree::parallel_for(dataStorage.size(), [&](embree::range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (dataStorage[i].first.candidate.signature.numSamples > PGL_SIGNATURE_MAX_SAMPLES)
                    dataStorage[i].first.candidate.signature.decay(0.5f);
            }
        });
        embree::parallel_for(candidateDataStorage.size(), [&](embree::range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (candidateDataStorage[i].signature.numSamples > PGL_SIGNATURE_MAX_SAMPLES)
                    candidateDataStorage[i].signature.decay(0.5f);
            }
        });
    }

    void prepareSampleReprojection(typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd, const SampleStatistics &stats) const {
        openpgl::Vector3 sampleVariance = stats.getVariance();
        float minDistance = length(sampleVariance);
        minDistance = 3.f * 3.f * sqrt(minDistance);

        for (auto it = samplesBegin; it != samplesEnd; ++it) {
            // Find the reprojected direction: nd = (pos + dist * dir - pivot).normalized()
            if (std::isinf(it->distance) || !(it->distance > 0.0f)) continue;

            const float distance = fmaxf(minDistance, it->distance);
            const openpgl::Point3 samplePosition(it->position.x, it->position.y, it->position.z);
            pgl_vec3f direction = it->direction;
            const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
            const openpgl::Point3 originPosition = samplePosition + sampleDirection * distance;
            openpgl::Vector3 newDirection = originPosition - stats.mean;
            const float newDistance = embree::length(newDirection);
            newDirection = newDirection / newDistance;

            pgl_vec3f reprojectedDirection = {newDirection[0], newDirection[1], newDirection[2]};
            it->reprojectedDirection = reprojectedDirection;
        }
    }

    void updateTreeNode(KDTree &kdTree, KDNode &node, uint8_t depth, const BBox &bounds,
                        TSamplesContainer &samples, const Range &sampleRange, TZeroValueSamplesContainer &zeroSamples, const Range &zeroSampleRange,
                        tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, CandidateRegionStorage &candidateDataStorage,
                        const Settings &settings, uint32_t iteration) const
    {
        OPENPGL_ASSERT(depth <= settings.maxDepth);
        uint8_t splitDim = 3;
        float splitPos;
        auto samplesBegin = samples.begin() + sampleRange.m_begin, samplesEnd = samples.begin() + sampleRange.m_end;
        auto zeroSamplesBegin = zeroSamples.begin() + zeroSampleRange.m_begin, zeroSamplesEnd = zeroSamples.begin() + zeroSampleRange.m_end;

        if (node.isLeaf()) {
            uint32_t dataIdx = node.getDataIdx();
            auto &[region, range] = dataStorage[dataIdx];
            auto &candidate = region.candidate;
            // Avoid double counting when this node is a first-level split
            SampleStatistics mergedStats = candidate.sampleStatistics;
            if (region.candidate.updated) {   // a result of a recent signature-based split
                region.regionBounds = bounds;
            } else {
                mergedStats.merge(computeStats(samplesBegin, samplesEnd));
            }

            KDNode *nodeLR[2] = {nullptr, nullptr};
            bool triggersSplit = false;

            if (depth + 1 <= settings.maxDepth && iteration < settings.initializingIters && mergedStats.getNumSamples() > settings.sampleCountThreshold) {
                // 1. sample count criterion
                triggersSplit = true;
                OPENPGL_ASSERT(!candidate.hasSplit());
                proposeSplit(mergedStats, splitDim, splitPos);
                OPENPGL_ASSERT(splitDim < 3);

                auto rDataItr = dataStorage.emplace_back(region, Range());
                RegionType *regionLR[2] = {&region, &rDataItr->first};

                // Inheritance
                for (uint8_t c: {0, 1}) {
                    regionLR[c]->candidate.sampleStatistics.clear();
                    initCandidateSignatures(0, regionLR[c]->candidate, candidateDataStorage, settings);
                    regionLR[c]->splitFlag = 1;
                    (c ? regionLR[c]->regionBounds.lower[splitDim] : regionLR[c]->regionBounds.upper[splitDim]) = splitPos;
                }

                // Extend KD tree
                uint32_t nodeIdLeft = kdTree.addChildrenPair();
                nodeLR[0] = &kdTree.getNode(nodeIdLeft);
                nodeLR[1] = &kdTree.getNode(nodeIdLeft + 1);
                node.setToInnerNode(splitDim, splitPos, nodeIdLeft);
                nodeLR[0]->setDataNodeIdx(dataIdx);
                nodeLR[1]->setDataNodeIdx(std::distance(dataStorage.begin(), rDataItr));
            } else if (depth + 1 <= settings.maxDepth && iteration >= settings.initializingIters) {
                // 2. Signature-based criterion
                if (settings.reproject)
                    prepareSampleReprojection(samplesBegin, samplesEnd, mergedStats);

                // Update candidate regions and check for promotion
                if (updateCandidateRegionsOnePromotion(depth, 0, candidate, candidate, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd, candidateDataStorage, settings)) {
                    OPENPGL_ASSERT(candidate.hasSplit());
                    splitDim = candidate.dim, splitPos = candidate.pivot;
                    triggersSplit = true;

                    auto rDataItr = dataStorage.emplace_back(region, Range());
                    RegionType *regionLR[2] = {&region, &rDataItr->first};
                    uint32_t lChildIdx = candidate.lChildIdx;

                    // Inheritance
                    for (uint8_t c: {0, 1}) {
                        regionLR[c]->candidate = candidateDataStorage[lChildIdx + c];
                        initCandidateSignatures(0, regionLR[c]->candidate, candidateDataStorage, settings);
                        regionLR[c]->splitFlag += 1;
                        // regionBounds set later
                    }
                    candidateDataStorage.recycle_pair(lChildIdx);

                    // Extend KD tree
                    uint32_t nodeIdLeft = kdTree.addChildrenPair();
                    nodeLR[0] = &kdTree.getNode(nodeIdLeft);
                    nodeLR[1] = &kdTree.getNode(nodeIdLeft + 1);
                    node.setToInnerNode(splitDim, splitPos, nodeIdLeft);
                    nodeLR[0]->setDataNodeIdx(dataIdx);
                    nodeLR[1]->setDataNodeIdx(std::distance(dataStorage.begin(), rDataItr));
                }
            }
            if (triggersSplit) {
                // Recurse into newly created children
                OPENPGL_ASSERT(nodeLR[0] != nullptr && nodeLR[1] != nullptr);
                auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, splitDim, splitPos);
                auto zeroSamplesMid = pivotSplitSamples(zeroSamplesBegin, zeroSamplesEnd, splitDim, splitPos);
                Range sampleRangeLR[2] = {
                    Range(sampleRange.m_begin, std::distance(samples.begin(), samplesMid)),
                    Range(std::distance(samples.begin(), samplesMid), sampleRange.m_end)
                };
                Range zeroSampleRangeLR[2] = {
                    Range(zeroSampleRange.m_begin, std::distance(zeroSamples.begin(), zeroSamplesMid)),
                    Range(std::distance(zeroSamples.begin(), zeroSamplesMid), zeroSampleRange.m_end)
                };
                auto boundsLR = splitBBox(bounds, splitDim, splitPos);

                invoke(
                    [&] { updateTreeNode(kdTree, *nodeLR[0], depth + 1, boundsLR.first, samples, sampleRangeLR[0], zeroSamples, zeroSampleRangeLR[0], dataStorage, candidateDataStorage, settings, iteration); },
                    [&] { updateTreeNode(kdTree, *nodeLR[1], depth + 1, boundsLR.second, samples, sampleRangeLR[1], zeroSamples, zeroSampleRangeLR[1], dataStorage, candidateDataStorage, settings, iteration); }
                );
            } else {
                // No split! Just merge in new samples
                if (!region.candidate.updated) {
                    region.candidate.sampleStatistics = mergedStats;
                }
                region.numZeroValueSamples = zeroSampleRange.size();
                // if (sampleRange.size() == 0) {
                //     std::cerr << "Warning: empty region at depth " << (int) depth << " id = " << dataIdx << " bounds = " << region.regionBounds << std::endl;
                // }
                range = sampleRange;
#ifdef OPENPGL_RADIANCE_CACHES
                range.m_is_begin = zeroSampleRange.m_begin;
                range.m_is_end = zeroSampleRange.m_end;
#endif
            }
        } else {
            // Internal node
            uint32_t nodeIdsLR[2];
            nodeIdsLR[0] = node.getLeftChildIdx(), nodeIdsLR[1] = nodeIdsLR[0] + 1;
            splitDim = node.getSplitDim(), splitPos = node.getSplitPivot();
            auto boundsLR = splitBBox(bounds, splitDim, splitPos);
            auto samplesMid = pivotSplitSamples(samples.begin() + sampleRange.m_begin, samples.begin() + sampleRange.m_end, splitDim, splitPos);
            auto zeroSamplesMid = pivotSplitSamples(zeroSamples.begin() + zeroSampleRange.m_begin, zeroSamples.begin() + zeroSampleRange.m_end, splitDim, splitPos);
            Range sampleRangesLR[2] = {Range(sampleRange.m_begin, std::distance(samples.begin(), samplesMid)),
                                       Range(std::distance(samples.begin(), samplesMid), sampleRange.m_end)};
            Range zeroSampleRangesLR[2] = {Range(zeroSampleRange.m_begin, std::distance(zeroSamples.begin(), zeroSamplesMid)),
                                           Range(std::distance(zeroSamples.begin(), zeroSamplesMid), zeroSampleRange.m_end)};

            invoke(
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[0]), depth + 1, boundsLR.first, samples, sampleRangesLR[0], zeroSamples, zeroSampleRangesLR[0], dataStorage, candidateDataStorage, settings, iteration); },
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[1]), depth + 1, boundsLR.second, samples, sampleRangesLR[1], zeroSamples, zeroSampleRangesLR[1], dataStorage, candidateDataStorage, settings, iteration); }
            );
        }
    }

    bool updateCandidateRegionsOnePromotion(uint8_t depth, uint8_t lookaheadLevel, const SubdivisionData &root, SubdivisionData &current,
        typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
        typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd,
        CandidateRegionStorage &candidateDataStorage, const Settings &settings) const {
        OPENPGL_ASSERT(depth <= settings.maxDepth);
        OPENPGL_ASSERT(lookaheadLevel <= settings.lookaheadDepth);

        auto update = [&settings, &root](SubdivisionData &region,
            typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
            typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd) {
            if (!region.updated) {
                if (!region.hasSplit())
                    region.sampleStatistics.merge(computeStats(samplesBegin, samplesEnd));
                region.updated = true;
            } else {
                OPENPGL_ASSERT(region.signature.numSamples == 0);
            }
            region.signature.addSamples(samplesBegin, samplesEnd);
            region.signature.addZeroSamples(std::distance(zeroSamplesBegin, zeroSamplesEnd));
        };

        // Update current
        if (lookaheadLevel == 0) {  // at the root
            update(current, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd);
        }

        // Lookahead
        if (!current.hasSplit()
            && lookaheadLevel + 1 <= settings.lookaheadDepth
            && depth + 1 <= settings.maxDepth && current.sampleStatistics.getNumSamples() >= settings.minSamplesCandidateSplit) {
            // Propose a new candidate split
            float splitPos;
            uint8_t splitDim;
            proposeSplit(current.sampleStatistics, splitDim, splitPos);
            current.setSplit(splitDim, splitPos);
            current.lChildIdx = candidateDataStorage.add_pair();
        }

        if (current.hasSplit()) {
            // Split samples
            auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, current.dim, current.pivot);
            auto zeroSamplesMid = pivotSplitSamples(zeroSamplesBegin, zeroSamplesEnd, current.dim, current.pivot);

            auto &left = candidateDataStorage[current.lChildIdx];
            auto &right = candidateDataStorage[current.lChildIdx + 1];

            // Update L/R signatures
            update(left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid);
            update(right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd);

            // Try promotion of the current split: either child should exceed the energy threshold
            if (checkPromotion(root, left, right, settings))
                return true;

            if (lookaheadLevel + 1 < settings.lookaheadDepth) {
                // Update L/R recursively
                bool hasPromotionLR[2] = {false, false};
                invoke(
                    [&] { hasPromotionLR[0] = updateCandidateRegionsOnePromotion(depth + 1, lookaheadLevel + 1, root, left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid, candidateDataStorage, settings); },
                    [&] { hasPromotionLR[1] = updateCandidateRegionsOnePromotion(depth + 1, lookaheadLevel + 1, root, right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd, candidateDataStorage, settings); }
                );
                return hasPromotionLR[0] || hasPromotionLR[1];
            }
        }

        return false;
    }

    // a: root, b: child
    static inline float getFluenceEnergy(const Signature &a, const Signature &b, const Settings &settings) {
        return Signature::getFluenceSplitConfidence(a, b, settings.signatureDistanceThreshold);
    }

    static float getAngularEnergy(const Signature &a, const Signature &b, const Settings &settings) {
        switch (settings.angularType)
        {
            case PGL_SPATIAL_ANGULAR_SERIES:
                return Signature::getAngularSplitConfidence(a, b, settings.angularDistanceThreshold);  // split probability
            case PGL_SPATIAL_ANGULAR_HEURISTIC:
                return Signature::getAngularDistance(a, b, settings.angularAlpha);  // effective angular distance
            default:
                return 0;
        }
    }

    static bool checkPromotion(const SubdivisionData &root, const SubdivisionData &left, const SubdivisionData &right, const Settings &settings) {
        if (!settings.enablePromotion) return false;
        float fluenceEnergy = std::max(getFluenceEnergy(root.signature, left.signature, settings), getFluenceEnergy(root.signature, right.signature, settings));
        float angularEnergy = 0;
        switch (settings.angularType) {
            case PGL_SPATIAL_ANGULAR_OFF:
                return left.signature.numSamples > settings.minSamplesPromotion && right.signature.numSamples > settings.minSamplesPromotion &&
                            // Signature
                            fluenceEnergy > settings.sufficientCriterionThreshold;
            case PGL_SPATIAL_ANGULAR_HEURISTIC:
                angularEnergy = std::max(getAngularEnergy(root.signature, left.signature, settings), getAngularEnergy(root.signature, right.signature, settings));
                return left.signature.numSamples > settings.minSamplesPromotion && right.signature.numSamples > settings.minSamplesPromotion &&
                       (
                            // Signature
                           fluenceEnergy > settings.sufficientCriterionThreshold ||
                           // Angular
                           angularEnergy > settings.angularDistanceThreshold
                       );
            case PGL_SPATIAL_ANGULAR_SERIES:
                angularEnergy = std::max(getAngularEnergy(root.signature, left.signature, settings), getAngularEnergy(root.signature, right.signature, settings));
                return left.signature.numSamples > settings.minSamplesPromotion && right.signature.numSamples > settings.minSamplesPromotion &&
                       (
                            // Signature
                           fluenceEnergy > settings.sufficientCriterionThreshold ||
                           // Angular
                           angularEnergy > 1.f - settings.angularAlpha
                       );
            default:
                std::cerr << "Unknown confidence type" << std::endl;
                return false;
        }
    }

    // Clear the signatures beneath current
    void initCandidateSignatures(int lookaheadLevel, SubdivisionData &current, CandidateRegionStorage &candidateDataStorage, const Settings &settings) const {
        current.signature.clear();
        if (current.hasSplit()) {
            auto &left = candidateDataStorage[current.lChildIdx];
            auto &right = candidateDataStorage[current.lChildIdx + 1];
            initCandidateSignatures(lookaheadLevel + 1, left, candidateDataStorage, settings);
            initCandidateSignatures(lookaheadLevel + 1, right, candidateDataStorage, settings);
        }
    }

    static std::pair<BBox, BBox> splitBBox(const BBox &bounds, uint8_t splitDim, float splitPos)
    {
        BBox left = bounds, right = bounds;
        right.lower[splitDim] = splitPos;
        left.upper[splitDim] = splitPos;
        return {left, right};
    }

    template<class TContainer, class FieldType>
    void evaluateRegionsNode(KDTree &kdTree, KDNode &node, uint8_t depth, TContainer &samples, const Range sampleRange, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, CandidateRegionStorage &candidateDataStorage, const Settings &buildSettings, const FieldType &field) const
    {
        if (sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {3};
        float splitPos = {0.0f};

        KDNode *nodesLeftRight[2] = {nullptr, nullptr};
        uint32_t dataIdx;
        Range sampleRangeLeftRight[2];

        if (node.isLeaf())
        {
            dataIdx = node.getDataIdx();
            TRegion &region = dataStorage[dataIdx].first;
            evaluateCandidateRegions<TContainer>(kdTree, region.candidate, region.candidate, samples.begin() + sampleRange.m_begin, samples.begin() + sampleRange.m_end, candidateDataStorage, buildSettings);
            return;
        }
        else
        {
            splitDim = node.getSplitDim();
            splitPos = node.getSplitPivot();
            uint32_t nodeIdLeft = node.getLeftChildIdx();
            nodesLeftRight[0] = &kdTree.getNode(nodeIdLeft);
            nodesLeftRight[1] = &kdTree.getNode(nodeIdLeft + 1);
        }

        OPENPGL_ASSERT(sampleRange.size() > 0);
        OPENPGL_ASSERT(splitDim < 3);

        size_t rPivotItr = 0;
#ifdef USE_EMBREE_PARALLEL
        rPivotItr = pivotSplitSamples2<typename TContainer::value_type>(samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos);
#else
        auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;
        auto it = pivotSplitSamples<TContainer>(begin, end, splitDim, splitPos);
        rPivotItr = std::distance(samples.begin(), it);
#endif

        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, rPivotItr);
        sampleRangeLeftRight[1] = Range(rPivotItr, sampleRange.m_end);

        invoke(
            [&] { evaluateRegionsNode(kdTree, *nodesLeftRight[0], depth + 1, samples, sampleRangeLeftRight[0], dataStorage, candidateDataStorage, buildSettings, field); },
            [&] { evaluateRegionsNode(kdTree, *nodesLeftRight[1], depth + 1, samples, sampleRangeLeftRight[1], dataStorage, candidateDataStorage, buildSettings, field); }
        );
    }

    template<class TContainer>
    void evaluateCandidateRegions(KDTree &kdTree, const SubdivisionData &root, SubdivisionData &current,
        typename TContainer::iterator samplesBegin, typename TContainer::iterator samplesEnd,
        CandidateRegionStorage &candidateDataStorage, const Settings &settings) const {
        constexpr bool isNonZeroSample = has_member_weight<typename TContainer::value_type>::value;

        // Update self
        if constexpr (isNonZeroSample) {
            current.signature.addSamples(samplesBegin, samplesEnd);
        } else {
            current.signature.addZeroSamples(std::distance(samplesBegin, samplesEnd));
        }

        if (current.hasSplit()) {
            // Split samples
            auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, current.dim, current.pivot);

            // Update L/R
            auto &leftRegion = candidateDataStorage[current.lChildIdx];
            auto &rightRegion = candidateDataStorage[current.lChildIdx + 1];

            evaluateCandidateRegions<TContainer>(kdTree, root, leftRegion, samplesBegin, samplesMid, candidateDataStorage, settings);
            evaluateCandidateRegions<TContainer>(kdTree, root, rightRegion, samplesMid, samplesEnd, candidateDataStorage, settings);
        }
    }

    template<typename TV>
    inline static uint8_t maxDimension(const TV& v) {
        return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
    };

    void proposeSplit(const SampleStatistics &stats, uint8_t &splitDim, float &splitPos) const
    {
        splitDim = maxDimension(stats.getVariance());
        splitPos = stats.getMean()[splitDim];
    }

    template<typename SampleIterator>
    static SampleStatistics computeStats(SampleIterator begin, SampleIterator end) {
        SampleStatistics sampleStats;
        for (auto itr = begin; itr != end; ++itr) {
            const Point3 position(itr->position.x, itr->position.y, itr->position.z);
            sampleStats.addSample(position);
        }
        return sampleStats;
    }

    template<typename TIterator>
    inline TIterator pivotSplitSamples(TIterator begin, TIterator end, uint8_t splitDimension, float pivot) const
    {
        auto pivotSplitPredicate = [splitDimension, pivot](auto &sample) -> bool {
            return sample.position[splitDimension] < pivot;
        };
        return std::partition(begin, end, pivotSplitPredicate);
    }

#ifdef USE_EMBREE_PARALLEL
    template <class DataType>
    inline size_t pivotSplitSamples2(DataType *samples, const size_t begin, const size_t end, uint8_t splitDimension, float pivot) const
    {
        auto isLeft = [&](const DataType &sample) {
            return Vector3(sample.position.x, sample.position.y, sample.position.z)[splitDimension] < pivot;
        };
        size_t center = 0;
        bool parallel = (end - begin) < PARALLEL_THRESHOLD ? false : true;
        if (!parallel)
        {
            center = embree::serial_partitioning(samples, begin, end, isLeft);
        }
        else
        {
            center = embree::parallel_partitioning(samples, begin, end, isLeft, PARALLEL_PARTITION_BLOCK_SIZE);
        }
        return center;
    }
#endif
};

template <class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline std::string KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder::Settings:" << std::endl;
    ss << "  maxDepth: " << maxDepth << std::endl;
    ss << "  minSamplesCandidateSplit: " << minSamplesCandidateSplit << std::endl;
    ss << "  minSamplesPromotion: " << minSamplesPromotion << std::endl;
    ss << "  sampleCountThreshold: " << sampleCountThreshold << std::endl;
    ss << "  initializingIters: " << initializingIters << std::endl;
    ss << "  lookaheadDepth: " << lookaheadDepth << std::endl;
    ss << "  signatureDistanceThreshold: " << signatureDistanceThreshold << std::endl;
    ss << "  sufficientCriterionThreshold: " << sufficientCriterionThreshold << std::endl;
    ss << "  angularDistanceThreshold: " << angularDistanceThreshold << std::endl;
    ss << "  angularAlpha: " << angularAlpha << std::endl;
    ss << "  knnJitterMultiplier: " << knnJitterMultiplier << std::endl;
    ss << "  reproject: " << reproject << std::endl;
    ss << "  enablePromotion: " << enablePromotion << std::endl;
    ss << "  angularType: " << angularType << std::endl;
    ss << "  knnType: " << knnType << std::endl;
    return ss.str();
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::serialize(std::ostream& stream)const
{
    stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(maxDepth));
    stream.write(reinterpret_cast<const char*>(&minSamplesCandidateSplit), sizeof(minSamplesCandidateSplit));
    stream.write(reinterpret_cast<const char*>(&minSamplesPromotion), sizeof(minSamplesPromotion));
    stream.write(reinterpret_cast<const char*>(&sampleCountThreshold), sizeof(sampleCountThreshold));
    stream.write(reinterpret_cast<const char*>(&initializingIters), sizeof(initializingIters));
    stream.write(reinterpret_cast<const char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.write(reinterpret_cast<const char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.write(reinterpret_cast<const char*>(&reproject), sizeof(reproject));
    stream.write(reinterpret_cast<const char*>(&enablePromotion), sizeof(enablePromotion));
    stream.write(reinterpret_cast<const char*>(&sufficientCriterionThreshold), sizeof(sufficientCriterionThreshold));
    stream.write(reinterpret_cast<const char*>(&angularDistanceThreshold), sizeof(angularDistanceThreshold));
    stream.write(reinterpret_cast<const char*>(&angularAlpha), sizeof(angularAlpha));
    stream.write(reinterpret_cast<const char*>(&knnJitterMultiplier), sizeof(knnJitterMultiplier));
    stream.write(reinterpret_cast<const char*>(&angularType), sizeof(angularType));
    stream.write(reinterpret_cast<const char*>(&knnType), sizeof(knnType));
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(maxDepth));
    stream.read(reinterpret_cast<char*>(&minSamplesCandidateSplit), sizeof(minSamplesCandidateSplit));
    stream.read(reinterpret_cast<char*>(&minSamplesPromotion), sizeof(minSamplesPromotion));
    stream.read(reinterpret_cast<char*>(&sampleCountThreshold), sizeof(sampleCountThreshold));
    stream.read(reinterpret_cast<char*>(&initializingIters), sizeof(initializingIters));
    stream.read(reinterpret_cast<char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.read(reinterpret_cast<char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.read(reinterpret_cast<char*>(&reproject), sizeof(reproject));
    stream.read(reinterpret_cast<char*>(&enablePromotion), sizeof(enablePromotion));
    stream.read(reinterpret_cast<char*>(&sufficientCriterionThreshold), sizeof(sufficientCriterionThreshold));
    stream.read(reinterpret_cast<char*>(&angularDistanceThreshold), sizeof(angularDistanceThreshold));
    stream.read(reinterpret_cast<char*>(&angularAlpha), sizeof(angularAlpha));
    stream.read(reinterpret_cast<char*>(&knnJitterMultiplier), sizeof(knnJitterMultiplier));
    stream.read(reinterpret_cast<char*>(&angularType), sizeof(angularType));
    stream.read(reinterpret_cast<char*>(&knnType), sizeof(knnType));
}

}

#undef THRESHOLD_VAR_RATIO
