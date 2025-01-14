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

#define THRESHOLD_VAR_RATIO         1e-4

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
    using Vector3d = embree::Vec3<double>;
    constexpr static double INF = std::numeric_limits<double>::infinity();
    mutable std::mutex mutex;
    mutable double signatureUpdateElapsed = 0;

#ifdef USE_EMBREE_PARALLEL
    static const size_t PARALLEL_THRESHOLD = 4 * 1024;
    static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 4 * 1024;
#endif

    struct Settings
    {
        PGL_SPATIAL_SPLIT_TYPE splitType {PGL_SPATIAL_SPLIT_BASELINE};
        uint32_t maxDepth {32};
        uint32_t minSamplesCandidateSplit {1000};  // to ensure the proposed split position is good enough
        uint32_t minSamplesPromotion {1000};  // to ensure the variance of signature estimates are small enough
        uint32_t sampleCountThreshold {PGL_TREE_MAX_SAMPLE_PER_LEAF};  // force a split if the number of samples exceeds this threshold and the depth is less than maxDepthSPLThreshold
        uint32_t maxDepthWithSampleCount {1};  // maximum depth with sample count threshold. Setting this to 1 means disabling it
        uint32_t lookaheadDepth {3};  // levels of lookahead
        float signatureDistanceThreshold {1.0f};  // triggers promotion if the distance between the signatures of the left and right children is greater than this threshold
        float decayRatio {0.25f};  // set from field
        float defensiveness {0.0f};  // the higher, the more likely to fall back to the baseline
        bool enablePromotion {true};
        float stdMultiplier {1.0f};
        float signatureDecay {1.0f};

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;

        bool operator==(const Settings &b) const
        {
            return splitType == b.splitType && maxDepth == b.maxDepth && minSamplesCandidateSplit == b.minSamplesCandidateSplit &&
                   minSamplesPromotion == b.minSamplesPromotion && sampleCountThreshold == b.sampleCountThreshold &&
                   maxDepthWithSampleCount == b.maxDepthWithSampleCount && lookaheadDepth == b.lookaheadDepth &&
                   signatureDistanceThreshold == b.signatureDistanceThreshold && decayRatio == b.decayRatio &&
                   defensiveness == b.defensiveness && enablePromotion == b.enablePromotion &&
                   stdMultiplier == b.stdMultiplier && signatureDecay == b.signatureDecay;
        }

        void updateFromConfig(const PGLKDTreeArguments &cfg)
        {
            maxDepth = cfg.maxDepth;
            minSamplesCandidateSplit = cfg.minSamplesCandidateSplit;
            minSamplesPromotion = cfg.minSamplesPromotion;
            sampleCountThreshold = cfg.sampleCountThreshold;
            maxDepthWithSampleCount = cfg.maxDepthWithSampleCount;
            lookaheadDepth = cfg.lookaheadDepth;
            signatureDistanceThreshold = cfg.signatureDistanceThreshold;
            stdMultiplier = cfg.stdMultiplier;
            signatureDecay = cfg.signatureDecay;
            enablePromotion = cfg.enablePromotion;
            decayRatio = cfg.ceDecay;
        }

        void loadToConfig(PGLKDTreeArguments &cfg) const
        {
            cfg.maxDepth = maxDepth;
            cfg.minSamplesCandidateSplit = minSamplesCandidateSplit;
            cfg.minSamplesPromotion = minSamplesPromotion;
            cfg.sampleCountThreshold = sampleCountThreshold;
            cfg.maxDepthWithSampleCount = maxDepthWithSampleCount;
            cfg.lookaheadDepth = lookaheadDepth;
            cfg.signatureDistanceThreshold = signatureDistanceThreshold;
            cfg.enablePromotion = enablePromotion;
            cfg.ceDecay = decayRatio;
        }
    };

    void build(KDTree &kdTree, const BBox &bounds, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<CandidateRegion> &candidateDataStorage, const Settings &buildSettings) const
    {
        std::cout << buildSettings.toString() << std::endl;

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;
        dataStorage[0].first.depth = 1;

        update(kdTree, samples, zeroSamples, dataStorage, candidateDataStorage, buildSettings, true);
    }

    void update(KDTree &kdTree, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<CandidateRegion> &candidateDataStorage, const Settings &buildSettings, bool isBuild = false) const
    {
        Timer timer;
        clock_t tic = clock();
        signatureUpdateElapsed = 0;
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.sampleCountThreshold+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        // Precompute bin index for samples
        embree::parallel_for(samples.size(), [&](embree::range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i)
                samples[i].binIndex = pgl_get_signature_index(samples[i].direction);
        });

        KDNode &root = kdTree.getRoot();
        BBox bounds;
        if (buildSettings.splitType == PGL_SPATIAL_SPLIT_PPG) {
            bounds = computeStats(samples.begin(), samples.end()).sampleBounds;  // here we don't use the kd tree bounds because it was 3x overestimated
            // Enlarge the bounds to make it a cube, see https://github.com/Tom94/practical-path-guiding/blob/fcf01afb436184e8a74bf300aa89f69b03ab25a2/mitsuba/src/integrators/path/guided_path.cpp#L855
            // This can avoid yielding of very long and thin reginos
            float half_width = reduce_max(bounds.size()) / 2;
            Vector3 center = bounds.center();
            bounds.lower = center - Vector3(half_width);
            bounds.upper = center + Vector3(half_width);
        } else {
            bounds = kdTree.getBounds();
        }
        std::cout << "Total bounds " << bounds << std::endl;

        updateTreeNode(kdTree, root, 1, 2, bounds, samples, Range(0, samples.size()), zeroSamples, Range(0, zeroSamples.size()), dataStorage, candidateDataStorage, buildSettings, isBuild);
        kdTree.finalize();
        double updateElapsed = timer.elapsed();
        clock_t toc = clock();
        double cpuTime = (double) (toc - tic) / CLOCKS_PER_SEC;
        std::cout << "KDTreePartitionBuilder::update() total update took " << updateElapsed * 1e-6 << " s, "
            << "total cpu time " << cpuTime << ", "
            << "percentage spent on signature update: " << signatureUpdateElapsed * 1e-6 / cpuTime * 100 << "%, "
            << "total valid regions: " << kdTree.getNumLeafs() << std::endl;
    }

    template<class TContainer, class FieldType>
    void evaluateRegions(KDTree &kdTree, TContainer &samples, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings, const FieldType &field) {
        KDNode &root = kdTree.getRoot();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth = 1;

        evaluateRegionsNode(&kdTree, &root, depth, samples, sampleRange, &dataStorage, buildSettings, field);
    }

    void updateTreeNode(KDTree &kdTree, KDNode &node, size_t depth, uint8_t prevSplitDim, const BBox &bounds,
        TSamplesContainer &samples, const Range &sampleRange, TZeroValueSamplesContainer &zeroSamples, const Range &zeroSampleRange,
        tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<CandidateRegion> &candidateDataStorage,
        const Settings &settings, bool isBuild) const
    {
        OPENPGL_ASSERT(depth <= settings.maxDepth);
        uint8_t splitDim = 3;
        float splitPos;
        auto samplesBegin = samples.begin() + sampleRange.m_begin, samplesEnd = samples.begin() + sampleRange.m_end;
        auto zeroSamplesBegin = zeroSamples.begin() + zeroSampleRange.m_begin, zeroSamplesEnd = zeroSamples.begin() + zeroSampleRange.m_end;

        if (node.isLeaf()) {
            uint32_t dataIdx = node.getDataIdx();
            auto &[region, range] = dataStorage[dataIdx];
            CandidateSplit &candidate = region.candidate;
            // Avoid double counting when this node is a first-level split
            SampleStatistics mergedStats = region.sampleStatistics;
            if (region.depth == 0) {   // a result of a recent signature-based split
                region.depth = depth;
                region.regionBounds = bounds;
            } else {
                mergedStats.merge(computeStats(samplesBegin, samplesEnd));
            }

            KDNode *nodeLR[2] = {nullptr, nullptr};
            bool hasSplit = false;

            // 1. sample count threshold
            if (depth + 1 <= settings.maxDepthWithSampleCount && mergedStats.getNumSamples() > settings.sampleCountThreshold) {
                hasSplit = true;
                if (candidate.valid()) {
                    splitDim = candidate.dim;
                    splitPos = candidate.pivot;
                } else splitBaseline(mergedStats, splitDim, splitPos);
                OPENPGL_ASSERT(splitDim < 3);

                auto rDataItr = dataStorage.emplace_back(region, Range());
                RegionType *regionLR[2] = {&region, &rDataItr->first};
                CandidateRegion *cregionLR[2] = {nullptr, nullptr};
                if (candidate.valid()) {
                    cregionLR[0] = &candidateDataStorage[candidate.lChildIdx];
                    cregionLR[1] = &candidateDataStorage[candidate.lChildIdx + 1];
                }

                for (uint8_t c: {0, 1}) {
                    if (candidate.valid()) regionLR[c]->sampleStatistics = cregionLR[c]->sampleStatistics;
                    else regionLR[c]->sampleStatistics.split(splitDim, splitPos, settings.decayRatio, c);
                    regionLR[c]->ceStatistics.decay(settings.decayRatio);
                    regionLR[c]->depth = depth + 1;
                    regionLR[c]->splitFlag = true;
                    (c ? regionLR[c]->regionBounds.lower[splitDim] : regionLR[c]->regionBounds.upper[splitDim]) = splitPos;
                }

                uint32_t nodeIdLeft = kdTree.addChildrenPair();
                nodeLR[0] = &kdTree.getNode(nodeIdLeft);
                nodeLR[1] = &kdTree.getNode(nodeIdLeft + 1);
                node.setToInnerNode(splitDim, splitPos, nodeIdLeft);
                nodeLR[0]->setDataNodeIdx(dataIdx);
                nodeLR[1]->setDataNodeIdx(std::distance(dataStorage.begin(), rDataItr));
            } else if (!isBuild && depth + 1 <= settings.maxDepth) {
                // 2. Signature splitting
                Timer timer;
                if (!candidate.valid() && mergedStats.getNumSamples() > settings.minSamplesCandidateSplit) {  // propose a new split
                    splitBaseline(mergedStats, splitDim, splitPos);
                    candidate.dim = splitDim, candidate.pivot = splitPos;
                    candidate.lChildIdx = std::distance(candidateDataStorage.begin(), candidateDataStorage.grow_by(2));
                    CandidateRegion *cregionLR[2] = {&candidateDataStorage[candidate.lChildIdx], &candidateDataStorage[candidate.lChildIdx + 1]};
                    for (uint8_t c: {0, 1}) {
                        cregionLR[c]->sampleStatistics = region.sampleStatistics;
                        cregionLR[c]->sampleStatistics.split(splitDim, splitPos, settings.decayRatio, c);
                    }
                }
                if (candidate.valid()) {
                    splitDim = candidate.dim, splitPos = candidate.pivot;
                    std::vector<std::pair<uint32_t, uint32_t>> newLeafs;
                    uint32_t leftNodeId = updateCandidateRegions(depth + 1, 1, kdTree, candidate, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd, candidateDataStorage, settings, newLeafs);
                    if (leftNodeId > 0) {
                        hasSplit = true;
                        // Allocate data for the new leafs
                        std::vector<uint32_t> dataInds(newLeafs.size());
                        dataInds[0] = dataIdx;  // reuse
                        auto firstdataItr = dataStorage.grow_by(newLeafs.size() - 1, {region, Range()});
                        for (int i = 1; i < newLeafs.size(); ++i) {
                            dataInds[i] = std::distance(dataStorage.begin(), firstdataItr) + i - 1;
                        }
                        for (int i = 0; i < newLeafs.size(); ++i) {
                            uint32_t newNodeId = newLeafs[i].first, canDataIdx = newLeafs[i].second;
                            KDNode &newNode = kdTree.getNode(newNodeId);
                            newNode.setDataNodeIdx(dataInds[i]);
                            CandidateRegion &canRegion = candidateDataStorage[canDataIdx];
                            RegionType &newRegion = dataStorage[dataInds[i]].first;
                            newRegion.sampleStatistics = canRegion.sampleStatistics;
                            newRegion.ceStatistics.decay(settings.decayRatio);
                            newRegion.splitFlag = true;
                            // Depth and regionBounds set later
                            newRegion.depth = 0;  // a special flag
                            newRegion.candidate = canRegion.candidate;
                        }

                        // Extend KD tree
                        node.setToInnerNode(splitDim, splitPos, leftNodeId);
                        nodeLR[0] = &kdTree.getNode(leftNodeId);
                        nodeLR[1] = &kdTree.getNode(leftNodeId + 1);
                    }
                }
                {
                    std::lock_guard guard(mutex);
                    signatureUpdateElapsed += timer.elapsed();
                }
            }
            if (hasSplit) {
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
                    [&] { updateTreeNode(kdTree, *nodeLR[0], depth + 1, splitDim, boundsLR.first, samples, sampleRangeLR[0], zeroSamples, zeroSampleRangeLR[0], dataStorage, candidateDataStorage, settings, isBuild); },
                    [&] { updateTreeNode(kdTree, *nodeLR[1], depth + 1, splitDim, boundsLR.second, samples, sampleRangeLR[1], zeroSamples, zeroSampleRangeLR[1], dataStorage, candidateDataStorage, settings, isBuild); }
                );
            } else {
                // No split! Just merge in new samples
                region.sampleStatistics = mergedStats;
                region.sampleStatistics.addNumZeroValueSamples(zeroSampleRange.size());
                region.numZeroValueSamples = zeroSampleRange.size();
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
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[0]), depth + 1, splitDim, boundsLR.first, samples, sampleRangesLR[0], zeroSamples, zeroSampleRangesLR[0], dataStorage, candidateDataStorage, settings, isBuild); },
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[1]), depth + 1, splitDim, boundsLR.second, samples, sampleRangesLR[1], zeroSamples, zeroSampleRangesLR[1], dataStorage, candidateDataStorage, settings, isBuild); }
            );
        }
    }

    // Update and try promotion recursively at the candidate nodes beneath candidate
    // Constructs the subtree and returns the kdNode idx if there is any promotion
    // Outputs new leaf nodes' (kd node id, candidate data idx) into newLeafs, in the DFS order
    uint32_t updateCandidateRegions(uint32_t depth, uint32_t lookaheadLevel, KDTree &kdTree, CandidateSplit &candidate,
        typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
        typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd,
        tbb::concurrent_vector<CandidateRegion> &candidateDataStorage, const Settings &settings, std::vector<std::pair<uint32_t, uint32_t>> &newLeafs) const {
        OPENPGL_ASSERT(depth <= settings.maxDepth);
        OPENPGL_ASSERT(lookaheadLevel >= 1 && lookaheadLevel <= settings.lookaheadDepth)
        OPENPGL_ASSERT(candidate.valid());
        OPENPGL_ASSERT(candidate.lChildIdx + 1 < candidateDataStorage.size());

        // Split samples
        auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, candidate.dim, candidate.pivot);
        auto zeroSamplesMid = pivotSplitSamples(zeroSamplesBegin, zeroSamplesEnd, candidate.dim, candidate.pivot);

        // Update L/R
        CandidateRegion &leftRegion = candidateDataStorage[candidate.lChildIdx];
        CandidateRegion &rightRegion = candidateDataStorage[candidate.lChildIdx + 1];

        leftRegion.sampleStatistics.merge(computeStats(samplesBegin, samplesMid));
        rightRegion.sampleStatistics.merge(computeStats(samplesMid, samplesEnd));

        leftRegion.signature.decay(settings.signatureDecay);
        leftRegion.signature.addSamples(samplesBegin, samplesMid);
        leftRegion.signature.addZeroSamples(std::distance(zeroSamplesBegin, zeroSamplesMid));

        rightRegion.signature.decay(settings.signatureDecay);
        rightRegion.signature.addSamples(samplesMid, samplesEnd);
        rightRegion.signature.addZeroSamples(std::distance(zeroSamplesMid, zeroSamplesEnd));

        candidate.energy = Signature::getDistance(leftRegion.signature, rightRegion.signature, settings.stdMultiplier);
        uint32_t lChildIdx = candidate.lChildIdx;
        uint8_t splitDim;

        // First try promotion at the current level
        if (settings.enablePromotion &&
            std::min(leftRegion.signature.getNumSamples(), rightRegion.signature.getNumSamples()) > settings.minSamplesPromotion &&
            candidate.energy > settings.signatureDistanceThreshold) {
            uint32_t leftNodeId = kdTree.addChildrenPair();
            KDNode &leftNode = kdTree.getNode(leftNodeId), &rightNode = kdTree.getNode(leftNodeId + 1);
            leftNode.setLeaf(), rightNode.setLeaf();  // The updateTreeNode method will fill in the missing data indices
            newLeafs.emplace_back(leftNodeId, lChildIdx);
            newLeafs.emplace_back(leftNodeId + 1, lChildIdx + 1);
            return leftNodeId;
        } else {
            // Then recurse into further levels, if they exist
            if (depth + 1 <= settings.maxDepth && lookaheadLevel + 1 <= settings.lookaheadDepth) {  // can still lookahead
                uint32_t leftLeftNodeId = 0, rightLeftNodeId = 0;
                if (!leftRegion.candidate.valid() && leftRegion.sampleStatistics.getNumSamples() >= settings.minSamplesCandidateSplit) {  // propose a new split
                    splitBaseline(leftRegion.sampleStatistics, splitDim, leftRegion.candidate.pivot);
                    leftRegion.candidate.dim = splitDim;
                    leftRegion.candidate.lChildIdx = std::distance(candidateDataStorage.begin(), candidateDataStorage.grow_by(2));
                }
                if (leftRegion.candidate.valid()) {
                    leftLeftNodeId = updateCandidateRegions(depth + 1, lookaheadLevel + 1, kdTree, leftRegion.candidate, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid, candidateDataStorage, settings, newLeafs);
                }

                if (!rightRegion.candidate.valid() && rightRegion.sampleStatistics.getNumSamples() >= settings.minSamplesCandidateSplit) {  // propose a new split
                    splitBaseline(rightRegion.sampleStatistics, splitDim, rightRegion.candidate.pivot);
                    rightRegion.candidate.dim = splitDim;
                    rightRegion.candidate.lChildIdx = std::distance(candidateDataStorage.begin(), candidateDataStorage.grow_by(2));
                }
                if (rightRegion.candidate.valid()) {
                    rightLeftNodeId = updateCandidateRegions(depth + 1, lookaheadLevel + 1, kdTree, rightRegion.candidate, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd, candidateDataStorage, settings, newLeafs);
                }

                if (leftLeftNodeId > 0 || rightLeftNodeId > 0) {  // create internal kd nodes if any promotion
                    uint32_t leftNodeId = kdTree.addChildrenPair();
                    KDNode &leftNode = kdTree.getNode(leftNodeId), &rightNode = kdTree.getNode(leftNodeId + 1);
                    if (leftLeftNodeId > 0) {
                        leftNode.setToInnerNode(leftRegion.candidate.dim, leftRegion.candidate.pivot, leftLeftNodeId);
                    } else {
                        leftNode.setLeaf();
                        newLeafs.emplace_back(leftNodeId, lChildIdx);
                    }
                    if (rightLeftNodeId > 0) {
                        rightNode.setToInnerNode(rightRegion.candidate.dim, rightRegion.candidate.pivot, rightLeftNodeId);
                    } else {
                        rightNode.setLeaf();
                        newLeafs.emplace_back(leftNodeId + 1, lChildIdx + 1);
                    }
                    return leftNodeId;
                }
                // else: no promotion
            } else {
                OPENPGL_ASSERT(!leftRegion.candidate.valid() && !rightRegion.candidate.valid());
            }
        }
        return 0;
    }

    static std::pair<BBox, BBox> splitBBox(const BBox &bounds, uint8_t splitDim, float splitPos)
    {
        BBox left = bounds, right = bounds;
        right.lower[splitDim] = splitPos;
        left.upper[splitDim] = splitPos;
        return {left, right};
    }

    template<class TContainer, class FieldType>
    void evaluateRegionsNode(KDTree *kdTree, KDNode *node, size_t depth, TContainer &samples, const Range sampleRange, tbb::concurrent_vector<std::pair<TRegion, Range> > *dataStorage, const Settings &buildSettings, const FieldType &field) const
    {
        OPENPGL_ASSERT(node != nullptr);
        using T = typename TContainer::value_type;
        constexpr bool isNonZeroSample = has_member_weight<T>::value;
        constexpr bool isSurfaceDist = has_function_applyCosineProduct<TSamplingDistribution>::value;
        if (sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {3};
        float splitPos = {0.0f};

        KDNode *nodesLeftRight[2] = {nullptr, nullptr};
        uint32_t dataIdx;
        Range sampleRangeLeftRight[2];

        if (node->isLeaf())
        {
            dataIdx = node->getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage->operator[](dataIdx);
            TRegion &region = regionAndRangeData.first;

            // Update CE for leaf regions without
            if constexpr (isNonZeroSample) {
                TSamplingDistribution guidingDist;
                // region.ceStatistics.self.decay(buildSettings.ceDecay);  // assume first update with nonzero samples
                // !! This can be slow
                for (size_t i = sampleRange.m_begin; i < sampleRange.m_end; i++) {
                    const T &sample = samples[i];
                    float weight = sample.weight;
                    // Evaluate pdf
                    const auto dist = &region.distribution;
                    Point3 position(sample.position.x, sample.position.y, sample.position.z);
                    auto _dir = pgl_vec3f(sample.direction);
                    Vector3 dir(_dir.x, _dir.y, _dir.z);
                    guidingDist.init(dist, position); // Applied parallax shift
                    if constexpr (isSurfaceDist) {
                        auto _normal = pgl_vec3f(sample.normal);
                        Vector3 normal(_normal.x, _normal.y, _normal.z);
                        guidingDist.applyCosineProduct(normal);
                    }
                    float pdf = guidingDist.pdf(dir);
                    // float pdf = sample.guidingPDF;
                    region.ceStatistics.addSample(weight, pdf);
                }
                // TODO
                // CandidateSplit &candidate = region.candidate;
                // if (candidate.valid()) {
                //     splitDim = candidate.dim;
                //     splitPos = candidate.pivot;
                //     // Split samples
                //     auto samplesMid = pivotSplitSamples(samples.begin() + sampleRange.m_begin, samples.begin() + sampleRange.m_end, splitDim, splitPos);
                //     // Update the LR signatures
                //     candidate.signaturesLR[0].decay(buildSettings.signatureDecay);
                //     candidate.signaturesLR[0].addSamples(samples.begin() + sampleRange.m_begin, samplesMid);
                //     candidate.signaturesLR[1].decay(buildSettings.signatureDecay);
                //     candidate.signaturesLR[1].addSamples(samplesMid, samples.begin() + sampleRange.m_end);
                // }
            } else {
                region.ceStatistics.addZeroWeightSamples(sampleRange.size());

                // for (uint8_t dim = 0; dim < 3; ++dim) {
                //     auto &candidate = region.candidateSplits[dim];
                //     if (!candidate.valid()) continue;
                //     splitDim = dim;
                //     splitPos = candidate.pos;
                //     // Split samples
                //     auto samplesMid = pivotSplitSamples(samples.begin() + sampleRange.m_begin, samples.begin() + sampleRange.m_end, splitDim, splitPos);
                //     // Update the LR signatures
                //     candidate.signaturesLR[0].addZeroSamples(std::distance(samples.begin() + sampleRange.m_begin, samplesMid));
                //     candidate.signaturesLR[1].addZeroSamples(std::distance(samplesMid, samples.begin() + sampleRange.m_end));
                // }
            }
            return;
        }
        else
        {
            splitDim = node->getSplitDim();
            splitPos = node->getSplitPivot();
            uint32_t nodeIdLeft = node->getLeftChildIdx();
            nodesLeftRight[0] = &kdTree->getNode(nodeIdLeft);
            nodesLeftRight[1] = &kdTree->getNode(nodeIdLeft + 1);
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
            [&] { evaluateRegionsNode(kdTree, nodesLeftRight[0], depth + 1, samples, sampleRangeLeftRight[0], dataStorage, buildSettings, field); },
            [&] { evaluateRegionsNode(kdTree, nodesLeftRight[1], depth + 1, samples, sampleRangeLeftRight[1], dataStorage, buildSettings, field); }
        );
    }

    // Compute the candidate split position, return the gain estimate that will guide when to split
    template<typename SampleIterator>
    inline float proposeSplit(uint8_t prevSplitDim, const BBox &bounds, SampleIterator begin, SampleIterator end, const SampleStatistics &stats,
                              uint8_t &splitDim, float &splitPos, const Settings &buildSettings) const {
        size_t minSamplesPerSide = buildSettings.minSamplesCandidateSplit / 2;
        switch (buildSettings.splitType) {
            case PGL_SPATIAL_SPLIT_BASELINE: splitBaseline(stats, splitDim, splitPos); return 0;
            case PGL_SPATIAL_SPLIT_ROUNDROBIN: splitRoundRobin(prevSplitDim, stats, splitDim, splitPos); return 0;
            case PGL_SPATIAL_SPLIT_PPG: splitPPG(prevSplitDim, bounds, stats, splitDim, splitPos); return 0;

            case PGL_SPATIAL_SPLIT_VS: return varianceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_COVS: return covarianceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_IGS: return informationGainScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_FS: return fluenceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);

            default: throw std::runtime_error("Unknown split type");
        }
    }

    void splitBaseline(const SampleStatistics &stats, uint8_t &splitDim, float &splitPos) const
    {
        splitDim = maxDimension(stats.getVariance());
        splitPos = stats.getMean()[splitDim];
    }

    void splitRoundRobin(uint8_t prevSplitDim, const SampleStatistics &stats, uint8_t &splitDim, float &splitPos) const
    {
        Vector3 var = stats.getVariance();
        float maxVar = reduce_max(var);
        splitDim = (prevSplitDim + 1) % 3;
#ifdef THRESHOLD_VAR_RATIO
        while (var[splitDim] < THRESHOLD_VAR_RATIO * maxVar) {
#else
            while (var[splitDim] < DEGENERATE_VAR_THRESHOLD) {
#endif
            splitDim = (splitDim + 1) % 3;  // skip dimensions with low variance
        }
        splitPos = stats.getMean()[splitDim];
    }

    void splitPPG(uint8_t prevSplitDim, const BBox &bounds, const SampleStatistics &stats, uint8_t &splitDim, float &splitPos) const
    {
        Vector3 var = stats.getVariance();
        float maxVar = reduce_max(var);
        splitDim = (prevSplitDim + 1) % 3;
#ifdef THRESHOLD_VAR_RATIO
        while (var[splitDim] < THRESHOLD_VAR_RATIO * maxVar) {
#else
            while (var[splitDim] < DEGENERATE_VAR_THRESHOLD) {
#endif
            splitDim = (splitDim + 1) % 3;  // skip dimensions with low variance
        }
        splitPos = bounds.center()[splitDim];
    }

    template <typename SampleIterator>
    float varianceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos) const
    {
        const size_t N = std::distance(begin, end);
        const auto posVariances = sampleStats.getVariance();
        const uint8_t maxVarDim = maxDimension(posVariances);
        const double maxPosVariance = posVariances[maxVarDim];
        if (N <= 2 * minSamplesPerSide) {  // insufficient samples to split: fallback to baseline
//            std::cout << "Insufficient samples to split: " << N << ", fallback to baseline" << std::endl;
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }

        // Find the pivot such that the total squared distances to the center of mass after splitting is minimized
        Vector3d costs(INF);
        Vector3 splits;

        auto pos = [](typename TSamplesContainer::iterator it) { return Vector3d(it->position.x, it->position.y, it->position.z); };

        // O(N log N)
        for (uint8_t dim = 0; dim < 3; ++dim) {
            if (posVariances[dim] < THRESHOLD_VAR_RATIO * maxPosVariance) {
                // std::cout << "Skipping dimension " << int(dim) << " due to low sample variance " << sampleVariances[dim] << std::endl;
                continue;
            }

            // Sort samples along the current dimension
            std::sort(begin, end, [dim](typename TSamplesContainer::value_type a, typename TSamplesContainer::value_type b) {
                return get(a.position, dim) < get(b.position, dim);
            });

            // Iterate pivot locations begin+m .. end-m
            //  and compute the first/second momentum on-line, O(N)

            // Welford's online algorithm for variance
            double lCnt = (double) minSamplesPerSide;
            Vector3d lMean, lM2;
            computeMoments(begin, begin + minSamplesPerSide, lMean, lM2);

            double rCnt = (double) (N - minSamplesPerSide);
            Vector3d rMean, rM2;
            computeMoments(begin + minSamplesPerSide, end, rMean, rM2);

            OPENPGL_ASSERT( (size_t) lCnt == minSamplesPerSide && (size_t) rCnt == N - minSamplesPerSide );

            // Find the best split with the minimum cost in the current dim
            double currMinCost = INF;
            SampleIterator currSplitItr;
            for (auto it = begin + minSamplesPerSide; it != end - minSamplesPerSide + 1; ++it) {
                // Try splitting at x => samples < x goes to left, samples >= x goes to right
                auto x = pos(it);
                double lCost = sum(lM2) / lCnt;
                double rCost = sum(rM2) / rCnt;
                double lFrac = lCnt / N, rFrac = rCnt / N;
                double cost = lFrac * lCost + rFrac * rCost;  // sum of weighted total variances
                if (cost < currMinCost) {
                    currMinCost = cost;
                    currSplitItr = it;
                }

                // Update statistics
                {
                    lCnt++;
                    auto delta = x - lMean;
                    lMean += delta / lCnt;
                    auto delta2 = x - lMean;
                    lM2 += delta * delta2;
                }

                // Right statistics are updated in a reversed manner
                {
                    auto delta2 = x - rMean;
                    rMean = (rCnt * rMean - x) / (rCnt - 1);
                    auto delta = x - rMean;
                    rM2 -= delta * delta2;
                    rCnt--;
                }
            }
            OPENPGL_ASSERT( (size_t) lCnt == N - minSamplesPerSide + 1 && (size_t) rCnt == minSamplesPerSide - 1 );

            OPENPGL_ASSERT(currMinCost < INF);
            costs[dim] = currMinCost;
            splits[dim] = (get(std::prev(currSplitItr)->position, dim) + get(currSplitItr->position, dim)) / 2.0f;
        }

        uint8_t bestDim = minDimension(costs);
        double minCost = costs[bestDim];
        double parentCost = sum(posVariances);
        double gain = parentCost - minCost;
        if (gain > defensiveness * parentCost) {
            splitDim = bestDim;
            splitPos = splits[splitDim];
            return (float) (gain / parentCost);
        } else {  // fall back to baseline
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }
    }

    template <typename SampleIterator>
    float covarianceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos) const
    {
        const size_t N = std::distance(begin, end);
        const auto posVariances = sampleStats.getVariance();
        const uint8_t maxVarDim = maxDimension(posVariances);
        const double maxPosVariance = posVariances[maxVarDim];
        if (N <= 2 * minSamplesPerSide) {  // insufficient samples to split: fallback to baseline
//            std::cout << "Insufficient samples to split: " << N << ", fallback to baseline" << std::endl;
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }

        Vector3d mean, variances, covariances;
        computeMoments(begin, end, mean, variances, covariances);
        variances /= (double) N, covariances /= (double) N;

        auto pos = [](typename TSamplesContainer::iterator it) { return Vector3d(it->position.x, it->position.y, it->position.z); };
        auto volume = [=](Vector3d v, const Vector3d &c) -> double {  // volume of the Gaussian ellipsoid with covariance matrix [v, c]
            v += Vector3d(1e-6);  // add a small value make covariance positive definite
            double det = 2 * reduce_mul(c) + reduce_mul(v) - c.y * c.y * v.x - c.z * c.z * v.y - c.x * c.x * v.z;
            return std::sqrt(std::max(det, 0.0));
        };

        // The lowest cost and the corresponding split position in each dimension
        Vector3d costs(INF);
        Vector3 splits;

        // O(N log N)
        for (uint8_t dim = 0; dim < 3; ++dim) {
            if (posVariances[dim] < THRESHOLD_VAR_RATIO * maxPosVariance) continue;  // skip dimensions with low variance

            // Sort samples along the current dimension
            std::sort(begin, end, [dim](typename TSamplesContainer::value_type a, typename TSamplesContainer::value_type b) {
                return get(a.position, dim) < get(b.position, dim);
            });

            // Iterate pivot locations begin+m .. end-m
            //  and compute the first/second/co- momentum on-line, O(N)

            double lCnt = (double) minSamplesPerSide;
            Vector3d lMean, lM2, lC;
            computeMoments(begin, begin + minSamplesPerSide, lMean, lM2, lC);

            double rCnt = (double) (N - minSamplesPerSide);
            Vector3d rMean, rM2, rC;
            computeMoments(begin + minSamplesPerSide, end, rMean, rM2, rC);

            OPENPGL_ASSERT( (size_t) lCnt == minSamplesPerSide && (size_t) rCnt == N - minSamplesPerSide );

            // Find the best split with the minimum cost in the current dim
            double currMinCost = INF;
            SampleIterator currSplitItr;
            for (auto it = begin + minSamplesPerSide; it != end - minSamplesPerSide + 1; ++it) {
                // Try splitting at x => samples < x goes to left, samples >= x goes to right
                auto x = pos(it);
                auto lVar = lM2 / lCnt, lCov = lC / lCnt;
                double lVol = volume(lVar, lCov);

                auto rVar = rM2 / rCnt, rCov = rC / rCnt;
                double rVol = volume(rVar, rCov);

                double lFrac = lCnt / N, rFrac = rCnt / N;
                double cost = lFrac * lVol + rFrac * rVol;  // weighted sum of volumes
                if (cost < currMinCost) {
                    currMinCost = cost;
                    currSplitItr = it;
                }

                // Update statistics
                {
                    lCnt++;
                    auto delta = x - lMean;
                    lMean += delta / lCnt;
                    auto delta2 = x - lMean;
                    lM2 += delta * delta2;
                    lC.x += delta.x * delta2.y;
                    lC.y += delta.y * delta2.z;
                    lC.z += delta.z * delta2.x;
                }

                // Right statistics are updated in a reversed manner
                {
                    auto delta2 = x - rMean;
                    rMean = (rCnt * rMean - x) / (rCnt - 1);
                    auto delta = x - rMean;
                    rM2 -= delta * delta2;
                    rC.x -= delta.x * delta2.y;
                    rC.y -= delta.y * delta2.z;
                    rC.z -= delta.z * delta2.x;
                    rCnt--;
                }
            }
            OPENPGL_ASSERT( (size_t) lCnt == N - minSamplesPerSide + 1 && (size_t) rCnt == minSamplesPerSide - 1 );

            costs[dim] = currMinCost;
            splits[dim] = (get(std::prev(currSplitItr)->position, dim) + get(currSplitItr->position, dim)) / 2.0f;
        }

        double parentCost = volume(variances, covariances);
        Vector3d gains = (Vector3d(parentCost) - costs) / Vector3d(parentCost);

        uint8_t maxDim = maxDimension(gains), midDim = midDimension(gains);
        double maxGain = gains[maxDim], midGain = gains[midDim];
        if (maxGain - midGain > defensiveness) {  // one axis is significantly better
            splitDim = maxDim;
        } else {  // fall back to baseline dimension, but still use the proposed pivot
            splitDim = maxVarDim;
        }
        splitPos = splits[splitDim];
        return (float) (gains[splitDim] / parentCost);  // never fall back to full baseline
    }

    template <typename SampleIterator>
    float informationGainScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos) const
    {
        const size_t N = std::distance(begin, end);
        const auto posVariances = sampleStats.getVariance();
        const uint8_t maxVarDim = maxDimension(posVariances);
        const double maxPosVariance = posVariances[maxVarDim];
        if (N <= 2 * minSamplesPerSide) {  // insufficient samples to split: fallback to baseline
//            std::cout << "Insufficient samples to split: " << N << ", fallback to baseline" << std::endl;
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }

        Vector3d mean, variances, covariances;
        computeMoments(begin, end, mean, variances, covariances);
        variances /= (double) N, covariances /= (double) N;

        auto pos = [](typename TSamplesContainer::iterator it) { return Vector3d(it->position.x, it->position.y, it->position.z); };
        auto entropy = [=](Vector3d v, const Vector3d &c) -> double {  // scaled entropy of the Gaussian distribution with covariance matrix [v, c]
            v += Vector3d(1e-6);  // add a small value make covariance positive definite
            double det = 2 * reduce_mul(c) + reduce_mul(v) - c.y * c.y * v.x - c.z * c.z * v.y - c.x * c.x * v.z;
            return 0.5 * std::log2(std::max(det, std::numeric_limits<double>::min()));
        };

        // The lowest cost and the corresponding split position in each dimension
        Vector3d costs(INF);
        Vector3 splits;

        // O(N log N)
        for (uint8_t dim = 0; dim < 3; ++dim) {
            if (posVariances[dim] < THRESHOLD_VAR_RATIO * maxPosVariance) continue;  // skip dimensions with low variance

            // Sort samples along the current dimension
            std::sort(begin, end, [dim](typename TSamplesContainer::value_type a, typename TSamplesContainer::value_type b) {
                return get(a.position, dim) < get(b.position, dim);
            });

            // Iterate pivot locations begin+m .. end-m
            //  and compute the first/second/co- momentum on-line, O(N)

            double lCnt = (double) minSamplesPerSide;
            Vector3d lMean, lM2, lC;
            computeMoments(begin, begin + minSamplesPerSide, lMean, lM2, lC);

            double rCnt = (double) (N - minSamplesPerSide);
            Vector3d rMean, rM2, rC;
            computeMoments(begin + minSamplesPerSide, end, rMean, rM2, rC);

            OPENPGL_ASSERT( (size_t) lCnt == minSamplesPerSide && (size_t) rCnt == N - minSamplesPerSide );

            // Find the best split with the minimum cost in the current dim
            double currMinCost = INF;
            SampleIterator currSplitItr;
            for (auto it = begin + minSamplesPerSide; it != end - minSamplesPerSide + 1; ++it) {
                // Try splitting at x => samples < x goes to left, samples >= x goes to right
                auto x = pos(it);
                auto lVar = lM2 / lCnt, lCov = lC / lCnt;
                double lEntropy = entropy(lVar, lCov);

                auto rVar = rM2 / rCnt, rCov = rC / rCnt;
                double rEntropy = entropy(rVar, rCov);

                double lFrac = lCnt / N, rFrac = rCnt / N;
                double cost = lFrac * lEntropy + rFrac * rEntropy;  // weighted sum of entropies
                if (cost < currMinCost) {
                    currMinCost = cost;
                    currSplitItr = it;
                }

                // Update statistics
                {
                    lCnt++;
                    auto delta = x - lMean;
                    lMean += delta / lCnt;
                    auto delta2 = x - lMean;
                    lM2 += delta * delta2;
                    lC.x += delta.x * delta2.y;
                    lC.y += delta.y * delta2.z;
                    lC.z += delta.z * delta2.x;
                }

                // Right statistics are updated in a reversed manner
                {
                    auto delta2 = x - rMean;
                    rMean = (rCnt * rMean - x) / (rCnt - 1);
                    auto delta = x - rMean;
                    rM2 -= delta * delta2;
                    rC.x -= delta.x * delta2.y;
                    rC.y -= delta.y * delta2.z;
                    rC.z -= delta.z * delta2.x;
                    rCnt--;
                }
            }
            OPENPGL_ASSERT( (size_t) lCnt == N - minSamplesPerSide + 1 && (size_t) rCnt == minSamplesPerSide - 1 );

            costs[dim] = currMinCost;
            splits[dim] = (get(std::prev(currSplitItr)->position, dim) + get(currSplitItr->position, dim)) / 2.0f;
        }

        double parentCost = entropy(variances, covariances);
        Vector3d gains = Vector3d(parentCost) - costs;

        uint8_t maxDim = maxDimension(gains), midDim = midDimension(gains);
        double maxGain = gains[maxDim], midGain = gains[midDim];
        if (maxGain - midGain > defensiveness) {  // one axis is significantly better (we use absolute difference for information gain)
            splitDim = maxDim;
        } else {  // fall back to baseline dimension, but still use the proposed pivot
            splitDim = maxVarDim;
        }
        splitPos = splits[splitDim];
        return (float) gains[splitDim];  // use the absolute gain
    }

    template <typename SampleIterator>
    float fluenceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos) const
    {
        const size_t N = std::distance(begin, end);
        const auto posVariances = sampleStats.getVariance();
        const uint8_t maxVarDim = maxDimension(posVariances);
        const double maxPosVariance = posVariances[maxVarDim];
        if (N <= 2 * minSamplesPerSide) {  // insufficient samples to split: fallback to baseline
//            std::cout << "Insufficient samples to split: " << N << ", fallback to baseline" << std::endl;
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }

        // Find the pivot such that the total variance of the sample weight after splitting is minimized
        Vector3d costs(INF);
        Vector3 splits;

        // O(N log N)
        for (uint8_t dim = 0; dim < 3; ++dim) {
            if (posVariances[dim] < THRESHOLD_VAR_RATIO * maxPosVariance) {
                // std::cout << "Skipping dimension " << int(dim) << " due to low sample variance " << sampleVariances[dim] << std::endl;
                continue;
            }

            // Sort samples along the current dimension
            std::sort(begin, end, [dim](typename TSamplesContainer::value_type a, typename TSamplesContainer::value_type b) {
                return get(a.position, dim) < get(b.position, dim);
            });

            // Iterate pivot locations begin+m .. end-m
            //  and compute the first/second momentum on-line, O(N)

            // Welford's online algorithm for variance
            double lCnt = (double) minSamplesPerSide, lMean, lM2;
            computeWeightMoments(begin, begin + minSamplesPerSide, lMean, lM2);

            double rCnt = (double) (N - minSamplesPerSide), rMean, rM2;
            computeWeightMoments(begin + minSamplesPerSide, end, rMean, rM2);

            OPENPGL_ASSERT( (size_t) lCnt == minSamplesPerSide && (size_t) rCnt == N - minSamplesPerSide );

            // Find the best split with the minimum cost in the current dim
            double currMinCost = INF;
            SampleIterator currSplitItr;
            for (auto it = begin + minSamplesPerSide; it != end - minSamplesPerSide + 1; ++it) {
                // Try splitting at x => samples < x goes to left, samples >= x goes to right
                double w = it->weight;
                double lVar = lM2 / lCnt;
                double rVar = rM2 / rCnt;
                double lFrac = lCnt / N, rFrac = rCnt / N;
                auto cost = lFrac * lVar + rFrac * rVar;  // sum of weighted variances
                if (cost < currMinCost) {
                    currMinCost = cost;
                    currSplitItr = it;
                }

                // Update statistics
                {
                    lCnt++;
                    double delta = w - lMean;
                    lMean += delta / lCnt;
                    double delta2 = w - lMean;
                    lM2 += delta * delta2;
                }

                // Right statistics are updated in a reversed manner
                {
                    double delta2 = w - rMean;
                    rMean = (rCnt * rMean - w) / (rCnt - 1);
                    double delta = w - rMean;
                    rM2 -= delta * delta2;
                    rCnt--;
                }
            }
            OPENPGL_ASSERT( (size_t) lCnt == N - minSamplesPerSide + 1 && (size_t) rCnt == minSamplesPerSide - 1 );

            OPENPGL_ASSERT(currMinCost < INF);
            costs[dim] = currMinCost;
            splits[dim] = (get(std::prev(currSplitItr)->position, dim) + get(currSplitItr->position, dim)) / 2.0f;
        }

        uint8_t bestDim = minDimension(costs);
        double minCost = costs[bestDim];
        double parentCost;
        {
            double mean, M2;
            computeWeightMoments(begin, end, mean, M2);
            parentCost = M2 / (double) N;
        }
        double gain = parentCost - minCost;
        if (gain > defensiveness * parentCost) {
            splitDim = bestDim;
            splitPos = splits[splitDim];
            return (float) (gain / parentCost);
        } else {
            splitDim = maxVarDim;
            splitPos = sampleStats.getMean()[splitDim];
            return 0;
        }
    }

    template<typename SampleIterator>
    inline void splitRandom(SampleIterator begin, SampleIterator end, uint8_t &splitDim, float &splitPos) const
    {
        const SampleStatistics sampleStats = computeStats(begin, end);
        const Vector3 sampleVariance = sampleStats.getVariance();
        const Point3 sampleMean = sampleStats.getMean();

        // splitDim = maxDimension(sampleVariance);  // baseline
        uint8_t minDim = minDimension(sampleVariance);
        float tmp = sqr(sampleMean);
        uint8_t rand = *reinterpret_cast<int*>(&tmp) % 2;  // pseudo random
        splitDim = (minDim + rand + 1) % 3;  // randomly select one of the two larger dimensions
        splitPos = sampleMean[splitDim];
    }

    std::string toString() const { return "KDTreePartitionBuilder\n"; }

    template<typename SampleIterator>
    inline SampleStatistics computeStats(SampleIterator begin, SampleIterator end) const {
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

    inline typename TSamplesContainer::iterator medianSplitSamples(typename TSamplesContainer::iterator begin, typename TSamplesContainer::iterator end, uint8_t splitDimension) const {
        auto medianItr = begin + (std::distance(begin, end) >> 1);
        std::nth_element(begin, medianItr, end, [splitDimension](typename TSamplesContainer::value_type a, typename TSamplesContainer::value_type b) {
            return get(a.position, splitDimension) < get(b.position, splitDimension);
        });
        return medianItr;
    }

    template<typename TV>
    inline static uint8_t maxDimension(const TV& v) {
        return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
    };

    template<typename TV>
    inline static uint8_t minDimension(const TV& v) {
        return v[v[1] < v[0]] < v[2] ? v[1] < v[0] : 2;
    };

    template<typename TV>
    inline static uint8_t midDimension(const TV& v) {
        if (v[0] < v[1]) {
            return v[1] < v[2] ? 1 : (v[0] < v[2] ? 2 : 0);
        } else {
            return v[0] < v[2] ? 0 : (v[1] < v[2] ? 2 : 1);
        }
    };

    inline static float get(pgl_vec3f p, uint8_t dim) {
        return (&p.x)[dim];
    }

    template <typename SampleIterator>
    static void computeMoments(const SampleIterator begin, const SampleIterator end, Vector3d &mean, Vector3d &M2) {
        double cnt = 0;
        mean = M2 = Vector3d(0);
        for (auto it = begin; it != end; ++it) {
            cnt++;
            auto x = Vector3d(it->position.x, it->position.y, it->position.z);
            auto delta = x - mean;
            mean += delta / cnt;
            auto delta2 = x - mean;
            M2 += delta * delta2;
        }
        OPENPGL_ASSERT( (size_t) cnt == std::distance(begin, end) );
    }

    template <typename SampleIterator>
    static void computeMoments(const SampleIterator begin, const SampleIterator end, Vector3d &mean, Vector3d &M2, Vector3d &C) {
        double cnt = 0;
        mean = M2 = C = Vector3d(0);
        for (auto it = begin; it != end; ++it) {
            cnt++;
            auto x = Vector3d(it->position.x, it->position.y, it->position.z);
            auto delta = x - mean;
            mean += delta / cnt;
            auto delta2 = x - mean;
            M2 += delta * delta2;
            C.x += delta.x * delta2.y;
            C.y += delta.y * delta2.z;
            C.z += delta.z * delta2.x;
        }
        OPENPGL_ASSERT( (size_t) cnt == std::distance(begin, end) );
    }

    template <typename SampleIterator>
    static void computeWeightMoments(const SampleIterator begin, const SampleIterator end, double &mean, double &M2) {
        double cnt = 0;
        mean = M2 = 0;
        for (auto it = begin; it != end; ++it) {
            cnt++;
            double w = it->weight;
            double delta = w - mean;
            mean += delta / cnt;
            double delta2 = w - mean;
            M2 += delta * delta2;
        }
        OPENPGL_ASSERT( (size_t) cnt == std::distance(begin, end) );
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
    ss << "  maxDepthWithSampleCount: " << maxDepthWithSampleCount << std::endl;
    ss << "  lookaheadDepth: " << lookaheadDepth << std::endl;
    ss << "  signatureDistanceThreshold: " << signatureDistanceThreshold << std::endl;
    ss << "  decayRatio: " << decayRatio << std::endl;
    ss << "  defensiveness: " << defensiveness << std::endl;
    ss << "  enablePromotion: " << enablePromotion << std::endl;

    return ss.str();
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::serialize(std::ostream& stream)const
{
    stream.write(reinterpret_cast<const char*>(&splitType), sizeof(splitType));
    stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(maxDepth));
    stream.write(reinterpret_cast<const char*>(&minSamplesCandidateSplit), sizeof(minSamplesCandidateSplit));
    stream.write(reinterpret_cast<const char*>(&minSamplesPromotion), sizeof(minSamplesPromotion));
    stream.write(reinterpret_cast<const char*>(&sampleCountThreshold), sizeof(sampleCountThreshold));
    stream.write(reinterpret_cast<const char*>(&maxDepthWithSampleCount), sizeof(maxDepthWithSampleCount));
    stream.write(reinterpret_cast<const char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.write(reinterpret_cast<const char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.write(reinterpret_cast<const char*>(&decayRatio), sizeof(decayRatio));
    stream.write(reinterpret_cast<const char*>(&defensiveness), sizeof(defensiveness));
    stream.write(reinterpret_cast<const char*>(&enablePromotion), sizeof(enablePromotion));
    stream.write(reinterpret_cast<const char*>(&stdMultiplier), sizeof(stdMultiplier));
    stream.write(reinterpret_cast<const char*>(&signatureDecay), sizeof(signatureDecay));
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&splitType), sizeof(splitType));
    stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(maxDepth));
    stream.read(reinterpret_cast<char*>(&minSamplesCandidateSplit), sizeof(minSamplesCandidateSplit));
    stream.read(reinterpret_cast<char*>(&minSamplesPromotion), sizeof(minSamplesPromotion));
    stream.read(reinterpret_cast<char*>(&sampleCountThreshold), sizeof(sampleCountThreshold));
    stream.read(reinterpret_cast<char*>(&maxDepthWithSampleCount), sizeof(maxDepthWithSampleCount));
    stream.read(reinterpret_cast<char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.read(reinterpret_cast<char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.read(reinterpret_cast<char*>(&decayRatio), sizeof(decayRatio));
    stream.read(reinterpret_cast<char*>(&defensiveness), sizeof(defensiveness));
    stream.read(reinterpret_cast<char*>(&enablePromotion), sizeof(enablePromotion));
    stream.read(reinterpret_cast<char*>(&stdMultiplier), sizeof(stdMultiplier));
    stream.read(reinterpret_cast<char*>(&signatureDecay), sizeof(signatureDecay));
}

}

#undef THRESHOLD_VAR_RATIO
