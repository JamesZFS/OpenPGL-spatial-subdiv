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
#define MIN_SAMPLES_PER_SIDE        16
#define GAMMA_CUT                   0.1

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

template <typename TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
struct KDTreePartitionBuilder
{
    const static PGL_SPATIAL_STRUCTURE_TYPE SPATIAL_STRUCTURE_TYPE = PGL_SPATIAL_STRUCTURE_KDTREE;

    typedef KDTree SpatialStructure;
    using RegionType = TRegion;
    using Vector3d = embree::Vec3<double>;
    constexpr static double INF = std::numeric_limits<double>::infinity();

#ifdef USE_EMBREE_PARALLEL
    static const size_t PARALLEL_THRESHOLD = 4 * 1024;
    static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 4 * 1024;
#endif

    struct Settings
    {
        PGL_SPATIAL_SPLIT_TYPE splitType {PGL_SPATIAL_SPLIT_BASELINE};
        size_t minSamples {100};
        size_t maxSamples {PGL_TREE_MAX_SAMPLE_PER_LEAF};  // force a split if the number of samples exceeds this threshold and the depth is less than maxDepthSPLThreshold
        size_t maxDepth {32};
        size_t maxDepthSPLThreshold {1};  // maximum depth with Samples-Per-Leaf threshold. Setting this to 1 means disabling it
        float decayRatio {0.25f};  // set from field
        float defensiveness {0.0f};  // the higher, the more likely to fall back to the baseline
        float maxDivReductionRate{0.05f}; // split if the parent's divergence - child's divergence goes beyond this
        float gainThreshold {1.0f};  // force a split if the gain is above this threshold
        bool enableCE {true};  // enable creation of lookahead children
        bool enablePromotion {true};
        bool failureDecay {true};
        bool singleSidePromotion {true};

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;

        bool operator==(const Settings& b) const {
            return splitType == b.splitType && minSamples == b.minSamples && maxSamples == b.maxSamples && maxDepth == b.maxDepth
                && maxDepthSPLThreshold == b.maxDepthSPLThreshold && decayRatio == b.decayRatio && defensiveness == b.defensiveness && maxDivReductionRate == b.maxDivReductionRate && gainThreshold == b.gainThreshold
                && enableCE == b.enableCE && enablePromotion == b.enablePromotion && failureDecay == b.failureDecay && singleSidePromotion == b.singleSidePromotion;
        }

        void updateFromConfig(const PGLKDTreeArguments &cfg)
        {
            minSamples = cfg.minSamples;
            maxSamples = cfg.maxSamples;
            maxDepth = cfg.maxDepth;
            maxDepthSPLThreshold = cfg.maxDepthWithSampleCount;
            enableCE = cfg.enableCE;
            enablePromotion = cfg.enablePromotion;
            failureDecay = cfg.failureDecay;
            singleSidePromotion = cfg.singleSidePromotion;
            maxDivReductionRate = cfg.ceThreshold;
            decayRatio = cfg.ceDecay;
        }
    };

    void build(KDTree &kdTree, const BBox &bounds, TSamplesContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {
        std::cout << buildSettings.toString() << std::endl;
        if (buildSettings.minSamples > buildSettings.maxSamples / 3) {
            throw std::runtime_error("KDTreePartitionBuilder::build() minSamples must be less than maxSamples/3");
        }

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;
        dataStorage[0].first.depth = 1;

        updateTree(kdTree, samples, dataStorage, buildSettings);
    }

    void updateTree(KDTree &kdTree, TSamplesContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {
        Timer timer;
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.maxSamples+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

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

        updateTreeNode(kdTree, root, 1, 2, bounds, samples, {0, samples.size()}, dataStorage, buildSettings);
        kdTree.finalize();
        double elapsedMicroSec = timer.elapsed();
        std::cout << "KDTreePartitionBuilder::update() took " << elapsedMicroSec * 1e-6 << " seconds, total valid regions: " << kdTree.getNumLeafs() << std::endl;
    }

    void insertTree(KDTree &kdTree, TZeroValueSamplesContainer &samples, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage) const
    {
        KDNode &root = kdTree.getRoot();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth = 1;

        insertTreeNode(&kdTree, root, depth, samples, sampleRange, &dataStorage);
    }

    template<class TContainer, class FieldType>
    void updateCEStats(KDTree &kdTree, TContainer &samples, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings, const FieldType &field) {
        KDNode &root = kdTree.getRoot();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth = 1;

        updateCEStatsNode(&kdTree, &root, depth, samples, sampleRange, &dataStorage, buildSettings, field);
    }

    void updateTreeNode(KDTree &kdTree, KDNode &node, size_t depth, uint8_t prevSplitDim, const BBox &bounds, TSamplesContainer &samples, const Range sampleRange, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &settings) const
    {
        const auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;  // for the current node
        if (node.isLeaf())
        {
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, Range> &regionRange = dataStorage[dataIdx];

            auto mergedStats = computeStats(begin, end);
            mergedStats.merge(regionRange.first.sampleStatistics);

            bool exceedSPLThreshold = depth < std::min(settings.maxDepth, settings.maxDepthSPLThreshold) && mergedStats.getNumSamples() > settings.maxSamples;
            bool canProposeSplit = depth < settings.maxDepth && mergedStats.getNumSamples() > 2 * settings.minSamples;  // so that a candidate split results in regions with sufficient training samples

            OPENPGL_ASSERT(!exceedSPLThreshold || canProposeSplit);  // if SPL threshold is exceeded, we must be able to propose a split

            if (regionRange.first.hasCandidateSplit()) {
                OPENPGL_ASSERT(canProposeSplit);
                // Update the candidate split and try promotion
                auto &candidate = regionRange.first.candidateSplit;
                OPENPGL_ASSERT(candidate.dataIdx + 1 < dataStorage.size());
                TRegion *regionLR[2] = {&dataStorage[candidate.dataIdx].first, &dataStorage[candidate.dataIdx + 1].first};
                OPENPGL_ASSERT(!regionLR[0]->hasCandidateSplit() && !regionLR[1]->hasCandidateSplit());

                // First update the divergence statistics of the children
                auto rPivotItr = pivotSplitSamples(begin, end, candidate.dim, candidate.pos);
                // field.updateAdaptiveMetrics(*regionLR[0], regionRange.first, begin, rPivotItr);
                // field.updateAdaptiveMetrics(*regionLR[1], regionRange.first, rPivotItr, end);

                if (settings.singleSidePromotion) {
                    // 1. Decide to promote or not
                    // 2. If yes, decide which side(s) should inherit from the parent
                    // 3. If not, decay the candidate children and update them
                    bool promoteLR[2] = {false, false};
                    // 1.
                    if (exceedSPLThreshold) {  // Possibly force a promotion with SPL threshold
                        promoteLR[0] = promoteLR[1] = true;
                    } else if (settings.enablePromotion) {  // Check if we should promote based on the divergence reduction
                        for (int i: {0, 1})
                            promoteLR[i] = regionLR[i]->ceStatistics.parent.getCE() - regionLR[i]->ceStatistics.self.getCE() > settings.maxDivReductionRate;
                    }
                    if (promoteLR[0] || promoteLR[1]) {
                        // 2.
                        for (int i: {0, 1}) {
                            regionLR[i]->unsetLookahead();
                            if (!promoteLR[i]) {
                                // Inherits the parent's distribution
                                regionLR[i]->distribution = regionRange.first.distribution;
                                regionLR[i]->trainingStatistics = regionRange.first.trainingStatistics;
#ifdef OPENPGL_RADIANCE_CACHES
                                regionLR[i]->outRadianceHist = regionRange.first.outRadianceHist;
#endif
                                regionLR[i]->splitFlag = true;

                                // Inherit CE stats to a tie position
                                regionLR[i]->ceStatistics.self = regionLR[i]->ceStatistics.parent;
                                regionLR[i]->decayDivergence(0);
                            }
                        }

                        // Extend KD tree
                        uint32_t nodeIdsLR[2];
                        nodeIdsLR[0] = kdTree.addChildrenPair();
                        nodeIdsLR[1] = nodeIdsLR[0] + 1;
                        node.setToInnerNode(candidate.dim, candidate.pos, nodeIdsLR[0]);
                        KDNode *nodeLR[2] = {&kdTree.getNode(nodeIdsLR[0]), &kdTree.getNode(nodeIdsLR[1])};
                        nodeLR[0]->setDataNodeIdx(candidate.dataIdx);
                        nodeLR[1]->setDataNodeIdx(candidate.dataIdx + 1);

                        // This region data is no-longer needed
                        // dataStorage.erase(dataIdx);
                        regionRange.first.removed = true;

                        // Recurse
                        Range sampleRangeLR[2] = {
                            Range(std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)),
                            Range(std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end))
                        };

                        BBox boundsLR[2] = {bounds, bounds};
                        boundsLR[0].upper[candidate.dim] = candidate.pos;
                        boundsLR[1].lower[candidate.dim] = candidate.pos;

                        tbb::parallel_invoke(
                            [&]{updateTreeNode(kdTree, *nodeLR[0], depth + 1, candidate.dim, boundsLR[0], samples, sampleRangeLR[0], dataStorage, settings);},
                            [&]{updateTreeNode(kdTree, *nodeLR[1], depth + 1, candidate.dim, boundsLR[1], samples, sampleRangeLR[1], dataStorage, settings);}
                        );
                    } else {
                        // 3.
                        if (settings.failureDecay) {
                            for (int i: {0, 1}) {
                                // regionLR[i]->sampleStatistics = regionRange.first.sampleStatistics;
                                // regionLR[i]->sampleStatistics.split(splitDim, splitPos, settings.decayRatio, (bool) i);
                                regionLR[i]->sampleStatistics.decay(settings.decayRatio);
                                // regionLR[i]->ceStatistics.parent = regionLR[i]->ceStatistics.self = regionRange.first.ceStatistics.self;
                                regionLR[i]->decayDivergence(settings.decayRatio);
                                regionLR[i]->splitFlag = true;
                                // regionBounds already adjusted
                            }
                        }

                        // Merge in new samples
                        regionRange.first.sampleStatistics = mergedStats;
                        regionRange.second = sampleRange;

                        // Update the lookahead children
                        regionLR[0]->sampleStatistics.merge(computeStats(begin, rPivotItr));
                        regionLR[1]->sampleStatistics.merge(computeStats(rPivotItr, end));
                        Range *sampleRangeLR[2] = {&dataStorage[candidate.dataIdx].second, &dataStorage[candidate.dataIdx + 1].second};

                        *sampleRangeLR[0] = {std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)};
                        *sampleRangeLR[1] = {std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end)};
                    }
                } else {
                    bool shouldPromote = false;
                    if (exceedSPLThreshold) {  // Possibly force a promotion with SPL threshold
                        shouldPromote = true;
                    } else if (settings.enablePromotion) {  // Check if we should promote based on the divergence reduction
                        float childEntropy = CEStatistics::weightedAverageCE(regionLR[0]->ceStatistics.self, regionLR[1]->ceStatistics.self);
                        float parentEntropy = CEStatistics::weightedAverageCE(regionLR[0]->ceStatistics.parent, regionLR[1]->ceStatistics.parent);
                        shouldPromote = parentEntropy - childEntropy > settings.maxDivReductionRate;
                    }
                    // if (!shouldPromote) {  // Last chance: re-propose a split position and check if we should promote based on the gain
                    //     uint8_t splitDim = -1;
                    //     float splitPos;
                    //     float gain = proposeSplit(prevSplitDim, bounds, begin, end, mergedStats, splitDim, splitPos, settings);
                    //     candidate.dim = splitDim;
                    //     candidate.pos = splitPos;
                    //     OPENPGL_ASSERT(candidate.dim < 3);
                    //     rPivotItr = pivotSplitSamples(begin, end, candidate.dim, candidate.pos);
                    //     regionLR[0]->regionBounds.upper[candidate.dim] = candidate.pos;
                    //     regionLR[1]->regionBounds.lower[candidate.dim] = candidate.pos;
                    //     shouldPromote = gain > settings.gainThreshold;
                    // }
                    if (shouldPromote) {
                        regionLR[0]->unsetLookahead();
                        regionLR[1]->unsetLookahead();
                        // Extend KD tree
                        uint32_t nodeIdsLR[2];
                        nodeIdsLR[0] = kdTree.addChildrenPair();
                        nodeIdsLR[1] = nodeIdsLR[0] + 1;
                        node.setToInnerNode(candidate.dim, candidate.pos, nodeIdsLR[0]);
                        KDNode *nodeLR[2] = {&kdTree.getNode(nodeIdsLR[0]), &kdTree.getNode(nodeIdsLR[1])};
                        nodeLR[0]->setDataNodeIdx(candidate.dataIdx);
                        nodeLR[1]->setDataNodeIdx(candidate.dataIdx + 1);

                        // This region data is no-longer needed
                        // dataStorage.erase(dataIdx);
                        regionRange.first.removed = true;

                        // Recurse
                        Range sampleRangeLR[2] = {
                            Range(std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)),
                            Range(std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end))
                    };

                        BBox boundsLR[2] = {bounds, bounds};
                        boundsLR[0].upper[candidate.dim] = candidate.pos;
                        boundsLR[1].lower[candidate.dim] = candidate.pos;

                        tbb::parallel_invoke(
                            [&]{updateTreeNode(kdTree, *nodeLR[0], depth + 1, candidate.dim, boundsLR[0], samples, sampleRangeLR[0], dataStorage, settings);},
                            [&]{updateTreeNode(kdTree, *nodeLR[1], depth + 1, candidate.dim, boundsLR[1], samples, sampleRangeLR[1], dataStorage, settings);}
                        );
                    } else {  // Decay the candidate children and update them
                        if (settings.failureDecay) {
                            for (int i: {0, 1}) {
                                // regionLR[i]->sampleStatistics = regionRange.first.sampleStatistics;
                                // regionLR[i]->sampleStatistics.split(splitDim, splitPos, settings.decayRatio, (bool) i);
                                regionLR[i]->sampleStatistics.decay(settings.decayRatio);
                                // regionLR[i]->ceStatistics.parent = regionLR[i]->ceStatistics.self = regionRange.first.ceStatistics.self;
                                regionLR[i]->decayDivergence(settings.decayRatio);
                                regionLR[i]->splitFlag = true;
                                // regionBounds already adjusted
                            }
                        }

                        // Merge in new samples
                        regionRange.first.sampleStatistics = mergedStats;
                        regionRange.second = sampleRange;

                        // Update the lookahead children
                        regionLR[0]->sampleStatistics.merge(computeStats(begin, rPivotItr));
                        regionLR[1]->sampleStatistics.merge(computeStats(rPivotItr, end));
                        Range *sampleRangeLR[2] = {&dataStorage[candidate.dataIdx].second, &dataStorage[candidate.dataIdx + 1].second};

                        *sampleRangeLR[0] = {std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)};
                        *sampleRangeLR[1] = {std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end)};
                    }
                }
            }
            else if (canProposeSplit) {
                uint8_t splitDim = -1;
                float splitPos;
                float gain = proposeSplit(prevSplitDim, bounds, begin, end, mergedStats, splitDim, splitPos, settings);
                OPENPGL_ASSERT(splitDim < 3);
                
                if (exceedSPLThreshold || gain > settings.gainThreshold) {  // Create a real split if SPL or gain threshold is exceeded
                    auto rPivotItr = pivotSplitSamples(begin, end, splitDim, splitPos);
                    // Left inherits self
                    auto rDataItr = dataStorage.push_back(regionRange);

                    regionRange.first.sampleStatistics.split(splitDim, splitPos, settings.decayRatio, false);
                    rDataItr->first.sampleStatistics.split(splitDim, splitPos, settings.decayRatio, true);

                    regionRange.first.ceStatistics.parent = rDataItr->first.ceStatistics.parent = regionRange.first.ceStatistics.self;  // initialize with a "tie"
                    regionRange.first.decayDivergence(0);
                    rDataItr->first.decayDivergence(0);

                    regionRange.first.depth = rDataItr->first.depth = depth + 1;

                    regionRange.first.splitFlag = true;
                    rDataItr->first.splitFlag = true;

                    regionRange.first.regionBounds.upper[splitDim] = splitPos;
                    rDataItr->first.regionBounds.lower[splitDim] = splitPos;

                    // Extend KD tree
                    uint32_t nodeIdsLR[2];
                    nodeIdsLR[0] = kdTree.addChildrenPair();
                    nodeIdsLR[1] = nodeIdsLR[0] + 1;
                    node.setToInnerNode(splitDim, splitPos, nodeIdsLR[0]);
                    KDNode *nodeLR[2] = {&kdTree.getNode(nodeIdsLR[0]), &kdTree.getNode(nodeIdsLR[1])};
                    nodeLR[0]->setDataNodeIdx(dataIdx);
                    nodeLR[1]->setDataNodeIdx(std::distance(dataStorage.begin(), rDataItr));

                    // Recurse
                    Range sampleRangeLR[2] = {
                            Range(std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)),
                            Range(std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end))
                    };

                    BBox boundsLR[2] = {bounds, bounds};
                    boundsLR[0].upper[splitDim] = splitPos;
                    boundsLR[1].lower[splitDim] = splitPos;

                    tbb::parallel_invoke(
                            [&]{updateTreeNode(kdTree, *nodeLR[0], depth + 1, splitDim, boundsLR[0], samples, sampleRangeLR[0], dataStorage, settings);},
                            [&]{updateTreeNode(kdTree, *nodeLR[1], depth + 1, splitDim, boundsLR[1], samples, sampleRangeLR[1], dataStorage, settings);}
                    );
                } else if (settings.enableCE) {  // Create a candidate split, resulting in two lookahead children
                    auto rPivotItr = pivotSplitSamples(begin, end, splitDim, splitPos);

                    // Copy current region to left/right child regions
                    auto lDataItr = dataStorage.grow_by(2);
                    regionRange.first.setCandidateSplit(splitDim, splitPos, std::distance(dataStorage.begin(), lDataItr));

                    TRegion *regionLR[2] = {&lDataItr->first, &std::next(lDataItr)->first};
                    Range *sampleRangeLR[2] = {&lDataItr->second, &std::next(lDataItr)->second};

                    // Child data handling
                    for (int i: {0, 1}) {
                        *regionLR[i] = regionRange.first;
                        regionLR[i]->sampleStatistics.split(splitDim, splitPos, settings.decayRatio, (bool) i);
                        regionLR[i]->ceStatistics.parent = regionRange.first.ceStatistics.self;  // initialize with a "tie"
                        regionLR[i]->decayDivergence(0);
                        regionLR[i]->splitFlag = true;
                        if (i == 0) {
                            regionLR[i]->regionBounds.upper[splitDim] = splitPos;
                        } else {
                            regionLR[i]->regionBounds.lower[splitDim] = splitPos;
                        }
                        regionLR[i]->setLookahead();
                        regionLR[i]->depth = depth + 1;
                    }

                    // Merge in new samples to the current region
                    regionRange.first.sampleStatistics = mergedStats;
                    regionRange.second = sampleRange;

                    // Merge in new samples to the lookahead children
                    regionLR[0]->sampleStatistics.merge(computeStats(begin, rPivotItr));
                    regionLR[1]->sampleStatistics.merge(computeStats(rPivotItr, end));

                    *sampleRangeLR[0] = {std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)};
                    *sampleRangeLR[1] = {std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end)};
                } else {
                    // No split. Just merge in new samples
                    regionRange.first.sampleStatistics = mergedStats;
                    regionRange.second = sampleRange;
                }
            }
            else {
                // No split. Just merge in new samples
                regionRange.first.sampleStatistics = mergedStats;
                regionRange.second = sampleRange;
            }
        }
        else  // Internal node
        {
            uint32_t nodeIdsLR[2];
            nodeIdsLR[0] = node.getLeftChildIdx();
            nodeIdsLR[1] = nodeIdsLR[0] + 1;

            KDNode *nodeLR[2] = {&kdTree.getNode(nodeIdsLR[0]), &kdTree.getNode(nodeIdsLR[1])};

            uint8_t splitDim = node.getSplitDim();
            float splitPos = node.getSplitPivot();

            BBox boundsLR[2] = {bounds, bounds};
            boundsLR[0].upper[splitDim] = splitPos;
            boundsLR[1].lower[splitDim] = splitPos;

            Range sampleRangeLR[2];
            auto rPivotItr = pivotSplitSamples(begin, end, splitDim, splitPos);  // [begin, rPivotItr) < splitPos, [rPivotItr, end) >= splitPos]

            sampleRangeLR[0] = {std::distance(samples.begin(), begin), std::distance(samples.begin(), rPivotItr)};
            sampleRangeLR[1] = {std::distance(samples.begin(), rPivotItr), std::distance(samples.begin(), end)};

            tbb::parallel_invoke(
                [&]{updateTreeNode(kdTree, *nodeLR[0], depth + 1, splitDim, boundsLR[0], samples, sampleRangeLR[0], dataStorage, settings);},
                [&]{updateTreeNode(kdTree, *nodeLR[1], depth + 1, splitDim, boundsLR[1], samples, sampleRangeLR[1], dataStorage, settings);}
            );
        }
    }

    void insertTreeNode(KDTree *kdTree, KDNode &node, size_t depth, TZeroValueSamplesContainer &samples, const Range sampleRange,
                        tbb::concurrent_vector<std::pair<TRegion, Range> > *dataStorage) const
    {
        if (sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        Range sampleRangeLeftRight[2];

        if (node.isLeaf())
        {
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage->operator[](dataIdx);
            regionAndRangeData.first.sampleStatistics.addNumZeroValueSamples(sampleRange.size());
            regionAndRangeData.first.numZeroValueSamples = sampleRange.size();
#ifdef OPENPGL_RADIANCE_CACHES
            regionAndRangeData.second.m_is_begin = sampleRange.m_begin;
            regionAndRangeData.second.m_is_end = sampleRange.m_end;
#endif
            return;
        }
        else
        {
            splitDim = node.getSplitDim();
            splitPos = node.getSplitPivot();
            nodeIdsLeftRight[0] = node.getLeftChildIdx();
            nodeIdsLeftRight[1] = nodeIdsLeftRight[0] + 1;
        }

        OPENPGL_ASSERT(!node.isLeaf());
        OPENPGL_ASSERT(sampleRange.size() > 0);

#ifdef USE_EMBREE_PARALLEL
        size_t rPivotItr = 0;
#else
        typename TZeroValueSamplesContainer::iterator rPivotItr;
        auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;
#endif
#ifdef USE_EMBREE_PARALLEL
        rPivotItr = pivotSplitSamples2<typename TZeroValueSamplesContainer::value_type>(samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos);
#else
        rPivotItr = pivotSplitSamples<TZeroValueSamplesContainer>(begin, end, splitDim, splitPos);
#endif

#ifdef USE_EMBREE_PARALLEL
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, rPivotItr);
        sampleRangeLeftRight[1] = Range(rPivotItr, sampleRange.m_end);
#else
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, std::distance(samples.begin(), rPivotItr));
        sampleRangeLeftRight[1] = Range(std::distance(samples.begin(), rPivotItr), sampleRange.m_end);
#endif
        /* This assert is a sanity check which is only valid with the assumption that the number of samples grows at same pace
           as the number of spatial nodes: in practice this is not the case (e.g., after many 1spp iterations)
        */
        // OPENPGL_ASSERT(sampleRangeLeftRight[0].size() > 1);
        // OPENPGL_ASSERT(sampleRangeLeftRight[1].size() > 1);

        tbb::parallel_invoke(
            [&] {
                insertTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, samples, sampleRangeLeftRight[0], dataStorage);
            },
            [&] {
                insertTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, samples, sampleRangeLeftRight[1], dataStorage);
            });
    }

    template<class TContainer, class FieldType>
    void updateCEStatsNode(KDTree *kdTree, KDNode *node, size_t depth, TContainer &samples, const Range sampleRange, tbb::concurrent_vector<std::pair<TRegion, Range> > *dataStorage, const Settings &buildSettings, const FieldType &field) const
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
        uint32_t dataIndsLeftRight[2];
        Range sampleRangeLeftRight[2];

        if (node->isLeaf())
        {
            dataIdx = node->getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage->operator[](dataIdx);
            TRegion &region = regionAndRangeData.first;
            if (region.hasCandidateSplit()) {
                splitDim = region.candidateSplit.dim;
                splitPos = region.candidateSplit.pos;
                dataIndsLeftRight[0] = region.candidateSplit.dataIdx;
                dataIndsLeftRight[1] = dataIndsLeftRight[0] + 1;
            }
            else {
                // Update CE for leaf regions without lookaheads
                if constexpr (isNonZeroSample) {
#if COMPUTE_CE_STYLE == 0
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
                        region.ceStatistics.self.addSample(weight, pdf);
                    }
#else
                    field.updateCE(region, samples.begin() + sampleRange.m_begin, samples.begin() + sampleRange.m_end);
#endif
                } else {
                    region.ceStatistics.self.addZeroWeightSamples(sampleRange.size());
                }
                return;
            }
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

        bool hasLookahead = nodesLeftRight[0] == nullptr || nodesLeftRight[1] == nullptr;

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

        if (hasLookahead) {
            // Update CE for lookaheads
            const auto &parentRegion = dataStorage->operator[](dataIdx).first;
            const auto parentDist = &parentRegion.distribution;
            for (int c: {0, 1}) {
                OPENPGL_ASSERT(nodesLeftRight[c] == nullptr);
                TRegion &childRegion = dataStorage->operator[](dataIndsLeftRight[c]).first;
                OPENPGL_ASSERT(childRegion.isLookahead);
                if constexpr (isNonZeroSample) {
#if COMPUTE_CE_STYLE == 0
                    TSamplingDistribution guidingDist;
                    // childRegion.ceStatistics.parent.decay(buildSettings.ceDecay);  // assume first update with nonzero samples
                    // childRegion.ceStatistics.self.decay(buildSettings.ceDecay);
                    // !! This can be slow
                    for (size_t i = sampleRangeLeftRight[c].m_begin; i < sampleRangeLeftRight[c].m_end; i++) {
                        const T &sample = samples[i];
                        float weight = sample.weight;
                        // Evaluate pdf
                        const auto childDist = &childRegion.distribution;
                        Point3 position(sample.position.x, sample.position.y, sample.position.z);
                        auto _dir = pgl_vec3f(sample.direction);
                        Vector3 dir(_dir.x, _dir.y, _dir.z);
                        auto _normal = pgl_vec3f(sample.normal);
                        Vector3 normal(_normal.x, _normal.y, _normal.z);

                        guidingDist.init(parentDist, position); // Applied parallax shift
                        if constexpr (isSurfaceDist)
                            guidingDist.applyCosineProduct(normal);
                        float qp = guidingDist.pdf(dir);
                        // float qp = sample.guidingPDF;
                        childRegion.ceStatistics.parent.addSample(weight, qp);

                        guidingDist.init(childDist, position); // Applied parallax shift
                        if constexpr (isSurfaceDist)
                            guidingDist.applyCosineProduct(normal);
                        float qc = guidingDist.pdf(dir);
                        childRegion.ceStatistics.self.addSample(weight, qc);
                    }
#else
                    field.updateCE(childRegion, parentRegion, samples.begin() + sampleRangeLeftRight[c].m_begin, samples.begin() + sampleRangeLeftRight[c].m_end);
#endif
                } else {
                    childRegion.ceStatistics.parent.addZeroWeightSamples(sampleRangeLeftRight[c].size());
                    childRegion.ceStatistics.self.addZeroWeightSamples(sampleRangeLeftRight[c].size());
                }
            }
        }
        else {
            tbb::parallel_invoke(
            [&] {
                updateCEStatsNode(kdTree, nodesLeftRight[0], depth + 1, samples, sampleRangeLeftRight[0], dataStorage, buildSettings, field);
            },
            [&] {
                updateCEStatsNode(kdTree, nodesLeftRight[1], depth + 1, samples, sampleRangeLeftRight[1], dataStorage, buildSettings, field);
            });
        }
    }

    // Compute the candidate split position, return the gain estimate that will guide when to split
    template<typename SampleIterator>
    inline float proposeSplit(uint8_t prevSplitDim, const BBox &bounds, SampleIterator begin, SampleIterator end, const SampleStatistics &stats,
                              uint8_t &splitDim, float &splitPos, const Settings &buildSettings) const {
        switch (buildSettings.splitType) {
            case PGL_SPATIAL_SPLIT_BASELINE: splitBaseline(stats, splitDim, splitPos); return 0;
            case PGL_SPATIAL_SPLIT_ROUNDROBIN: splitRoundRobin(prevSplitDim, stats, splitDim, splitPos); return 0;
            case PGL_SPATIAL_SPLIT_PPG: splitPPG(prevSplitDim, bounds, stats, splitDim, splitPos); return 0;

            case PGL_SPATIAL_SPLIT_VS: return varianceScan(begin, end, stats, buildSettings.minSamples, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_COVS: return covarianceScan(begin, end, stats, buildSettings.minSamples, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_IGS: return informationGainScan(begin, end, stats, buildSettings.minSamples, buildSettings.defensiveness, splitDim, splitPos);
            case PGL_SPATIAL_SPLIT_FS: return fluenceScan(begin, end, stats, buildSettings.minSamples, buildSettings.defensiveness, splitDim, splitPos);

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

    inline typename TSamplesContainer::iterator pivotSplitSamples(typename TSamplesContainer::iterator begin, typename TSamplesContainer::iterator end,
                                                           uint8_t splitDimension, float pivot) const
    {
        std::function<bool(typename TSamplesContainer::value_type)> pivotSplitPredicate
                = [splitDimension, pivot](typename TSamplesContainer::value_type sample) -> bool
                {
                    const Point3 position(sample.position.x, sample.position.y, sample.position.z);
                    return position[splitDimension] < pivot;

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
    ss << "  splitType: " << splitType << std::endl;
    ss << "  minSamples: " << minSamples << std::endl;
    ss << "  maxSamples: " << maxSamples << std::endl;
    ss << "  maxDepth: " << maxDepth << std::endl;
    ss << "  maxDepthSPLThreshold: " << maxDepthSPLThreshold << std::endl;
    ss << "  decayRatio: " << decayRatio << std::endl;
    ss << "  defensiveness: " << defensiveness << std::endl;
    ss << "  maxDivReductionRate: " << maxDivReductionRate << std::endl;
    ss << "  gainThreshold: " << gainThreshold << std::endl;
    ss << "  enableCE: " << enableCE << std::endl;

    return ss.str();
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::serialize(std::ostream& stream)const
{
    stream.write(reinterpret_cast<const char*>(&splitType), sizeof(PGL_SPATIAL_SPLIT_TYPE));
    stream.write(reinterpret_cast<const char*>(&minSamples), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&maxSamples), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&maxDepthSPLThreshold), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&decayRatio), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&defensiveness), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&maxDivReductionRate), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&gainThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&enableCE), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&enablePromotion), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&failureDecay), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&singleSidePromotion), sizeof(bool));
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&splitType), sizeof(PGL_SPATIAL_SPLIT_TYPE));
    stream.read(reinterpret_cast<char*>(&minSamples), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&maxSamples), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&maxDepthSPLThreshold), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&decayRatio), sizeof(float));
    stream.read(reinterpret_cast<char*>(&defensiveness), sizeof(float));
    stream.read(reinterpret_cast<char*>(&maxDivReductionRate), sizeof(float));
    stream.read(reinterpret_cast<char*>(&gainThreshold), sizeof(float));
    stream.read(reinterpret_cast<char*>(&enableCE), sizeof(bool));
    stream.read(reinterpret_cast<char*>(&enablePromotion), sizeof(bool));
    stream.read(reinterpret_cast<char*>(&failureDecay), sizeof(bool));
    stream.read(reinterpret_cast<char*>(&singleSidePromotion), sizeof(bool));
}

}

#undef THRESHOLD_VAR_RATIO
#undef MIN_SAMPLES_PER_SIDE
#undef GAMMA_CUT
#undef SINGLE_SIDE_PROMOTION
