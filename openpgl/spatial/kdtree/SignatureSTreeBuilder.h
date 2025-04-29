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
    using Vector3d = embree::Vec3<double>;
    constexpr static double INF = std::numeric_limits<double>::infinity();

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
        uint32_t sampleCountThreshold {PGL_TREE_MAX_SAMPLE_PER_LEAF};  // threshold of OpenPGL's standard subdivision scheme
        uint32_t forcedSampleCountThreshold {(uint32_t) -1};  // sample count to force a split during the signature splitting stage in the first iteration
        uint32_t initializingIters {1};  // the number of iterations to use the standard subdivision scheme, after which the signature threshold kicks in
        uint32_t lookaheadDepth {3};  // levels of lookahead
        float signatureDistanceThreshold {0.05f};  // triggers promotion if the distance between the signatures of the left and right children is greater than this threshold
        float decayRatio {0.25f};  // set from field
        float defensiveness {0.0f};  // the higher, the more likely to fall back to the baseline
        bool enablePromotion {true};
        float stdMultiplier {1.0f};
        float riskTolerance {0.1f}; // reject the signature subdivision if std / mean is above this threshold
        float tValueThreshold {3.0f}; // reject the signature subdivision if std / mean is above this threshold
        float inlierPercent {0.99f};  // filter outlier samples for signature computation
        float DBORstdMultiplier {3.0f};  // remove outliers that are outside [0, mu + DBORstdMultiplier * std]
        float tEpsK {0.0f};
        float varianceThreshold {1e-4f};  // skip the axis whose variance / max variance is less than this threshold
        bool multiplyCosine {false};  // whether to incorporate cosine terms into directional signatures
        bool reproject {false};  // whether to reproject samples to the center of the parent region when calculating signatures
        bool nonRecursive {false};  // if enabled, stop the subdivision when the promotion finishes
        bool singlePromotion {false}; // if enabled, promote at most one level for each parent node during the promotion handling
        bool optimizeSignature {false};  // if true, optimize signature computation by caching basis function values TODO: not compatible with signature ensemble
        PGL_SPATIAL_CONFIDENCE_TYPE confidenceType {PGL_SPATIAL_CONFIDENCE_NONE};  // confidence metric to tell us whether we can trust our difference metric
        PGL_SPATIAL_DEFENSIVE_TYPE defensiveType {PGL_SPATIAL_DEFENSIVE_FIXED};  // how to grow the defensive sample count threshold
        PGL_SPATIAL_FILTER_TYPE filterType {PGL_SPATIAL_FILTER_NONE};  // how to perform outlier removal
        std::vector<SignatureArguments> signatureEnsembleConfig;

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;

        bool operator==(const Settings &b) const
        {
            return splitType == b.splitType && maxDepth == b.maxDepth && minSamplesCandidateSplit == b.minSamplesCandidateSplit &&
                   minSamplesPromotion == b.minSamplesPromotion && sampleCountThreshold == b.sampleCountThreshold && forcedSampleCountThreshold == b.forcedSampleCountThreshold &&
                   initializingIters == b.initializingIters && lookaheadDepth == b.lookaheadDepth &&
                   signatureDistanceThreshold == b.signatureDistanceThreshold && decayRatio == b.decayRatio &&
                   defensiveness == b.defensiveness && enablePromotion == b.enablePromotion &&
                   stdMultiplier == b.stdMultiplier && riskTolerance == b.riskTolerance && tValueThreshold == b.tValueThreshold &&
                   inlierPercent == b.inlierPercent && DBORstdMultiplier == b.DBORstdMultiplier && tEpsK == b.tEpsK && varianceThreshold == b.varianceThreshold &&
                   multiplyCosine == b.multiplyCosine && reproject == b.reproject && nonRecursive == b.nonRecursive && singlePromotion == b.singlePromotion && optimizeSignature == b.optimizeSignature &&
                   confidenceType == b.confidenceType && defensiveType == b.defensiveType && filterType == b.filterType && signatureEnsembleConfig == b.signatureEnsembleConfig;
        }

        void updateFromConfig(const PGLKDTreeArguments &cfg)
        {
            splitType = cfg.splitType;
            maxDepth = cfg.maxDepth;
            minSamplesCandidateSplit = cfg.minSamplesCandidateSplit;
            minSamplesPromotion = cfg.minSamplesPromotion;
            sampleCountThreshold = cfg.sampleCountThreshold;
            forcedSampleCountThreshold = cfg.forcedSampleCountThreshold;
            initializingIters = cfg.initializingIters;
            lookaheadDepth = cfg.lookaheadDepth;
            signatureDistanceThreshold = cfg.signatureDistanceThreshold;
            stdMultiplier = cfg.stdMultiplier;
            riskTolerance = cfg.riskTolerance;
            tValueThreshold = cfg.tValueThreshold;
            inlierPercent = cfg.inlierPercent;
            DBORstdMultiplier = cfg.DBORstdMultiplier;
            enablePromotion = cfg.enablePromotion;
            decayRatio = cfg.ceDecay;
            tEpsK = cfg.tEpsK;
            varianceThreshold = cfg.varianceThreshold;
            multiplyCosine = cfg.multiplyCosine;
            reproject = cfg.reproject;
            nonRecursive = cfg.nonRecursive;
            singlePromotion = cfg.singlePromotion;
            optimizeSignature = cfg.optimizeSignature;
            confidenceType = cfg.confidenceType;
            defensiveType = cfg.defensiveType;
            filterType = cfg.filterType;
            signatureEnsembleConfig = cfg.signatureEnsembleConfig;
        }

        void loadToConfig(PGLKDTreeArguments &cfg) const
        {
            cfg.splitType = splitType;
            cfg.maxDepth = maxDepth;
            cfg.minSamplesCandidateSplit = minSamplesCandidateSplit;
            cfg.minSamplesPromotion = minSamplesPromotion;
            cfg.sampleCountThreshold = sampleCountThreshold;
            cfg.forcedSampleCountThreshold = forcedSampleCountThreshold;
            cfg.initializingIters = initializingIters;
            cfg.lookaheadDepth = lookaheadDepth;
            cfg.signatureDistanceThreshold = signatureDistanceThreshold;
            cfg.stdMultiplier = stdMultiplier;
            cfg.riskTolerance = riskTolerance;
            cfg.tValueThreshold = tValueThreshold;
            cfg.inlierPercent = inlierPercent;
            cfg.DBORstdMultiplier = DBORstdMultiplier;
            cfg.tEpsK = tEpsK;
            cfg.varianceThreshold = varianceThreshold;
            cfg.enablePromotion = enablePromotion;
            cfg.ceDecay = decayRatio;
            cfg.multiplyCosine = multiplyCosine;
            cfg.reproject = reproject;
            cfg.nonRecursive = nonRecursive;
            cfg.singlePromotion = singlePromotion;
            cfg.optimizeSignature = optimizeSignature;
            cfg.confidenceType = confidenceType;
            cfg.defensiveType = defensiveType;
            cfg.filterType = filterType;
            cfg.signatureEnsembleConfig = signatureEnsembleConfig;
        }
    };

    void build(KDTree &kdTree, const BBox &bounds, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &buildSettings, uint32_t iteration) const
    {
        std::cout << buildSettings.toString() << std::endl;

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;
        dataStorage[0].first.candidate.depth = 1;

        update(kdTree, samples, zeroSamples, dataStorage, candidateDataStorage, buildSettings, iteration);
    }

    void update(KDTree &kdTree, TSamplesContainer &samples, TZeroValueSamplesContainer &zeroSamples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &buildSettings, uint32_t iteration) const
    {
        Timer timer;
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.sampleCountThreshold+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        // if (iteration >= buildSettings.initializingIters && (!buildSettings.reproject && buildSettings.optimizeSignature))
        //     Signature::computeSampleBasisFunctions(samples.begin(), samples.end(), buildSettings.basisType);

        KDNode &root = kdTree.getRoot();
        BBox bounds;
        // if (buildSettings.splitType == PGL_SPATIAL_SPLIT_PPG) {
        //     bounds = computeStats(samples.begin(), samples.end()).sampleBounds;  // here we don't use the kd tree bounds because it was 3x overestimated
        //     // Enlarge the bounds to make it a cube, see https://github.com/Tom94/practical-path-guiding/blob/fcf01afb436184e8a74bf300aa89f69b03ab25a2/mitsuba/src/integrators/path/guided_path.cpp#L855
        //     // This can avoid yielding of very long and thin reginos
        //     float half_width = reduce_max(bounds.size()) / 2;
        //     Vector3 center = bounds.center();
        //     bounds.lower = center - Vector3(half_width);
        //     bounds.upper = center + Vector3(half_width);
        // } else {
        //     bounds = kdTree.getBounds();
        // }
        bounds = kdTree.getBounds();
        std::cout << "Total bounds " << bounds << std::endl;

        updateTreeNode(kdTree, root, 1, 2, bounds, samples, Range(0, samples.size()), zeroSamples, Range(0, zeroSamples.size()), dataStorage, candidateDataStorage, buildSettings, iteration);

        if (iteration >= buildSettings.initializingIters) {
            // Postprocessing: clear flags, decay signature when a region has way too many samples
            embree::parallel_for(dataStorage.size(), [&](embree::range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    dataStorage[i].first.candidate.updated = false;
                    if (dataStorage[i].first.candidate.signatures.getNumSamples() > PGL_SIGNATURE_MAX_SAMPLES)
                        dataStorage[i].first.candidate.signatures.decay(0.5f);
                }
            });
            embree::parallel_for(candidateDataStorage.size(), [&](embree::range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    candidateDataStorage[i].updated = false;
                    if (candidateDataStorage[i].signatures.getNumSamples() > PGL_SIGNATURE_MAX_SAMPLES)
                        candidateDataStorage[i].signatures.decay(0.5f);
                }
            });
        }
        kdTree.finalize();
        double updateElapsed = timer.elapsed();
        std::cout << "KDTreePartitionBuilder::update() total update took " << updateElapsed * 1e-6 << " s, "
            << "total valid regions: " << kdTree.getNumLeafs() << std::endl;
    }

    template<class TContainer, class FieldType>
    void evaluateRegions(KDTree &kdTree, TContainer &samples, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &buildSettings, const FieldType &field) {
        constexpr bool isNonZeroSample = has_member_weight<typename TContainer::value_type>::value;
        if constexpr(isNonZeroSample) {
            // if (buildSettings.optimizeSignature) Signature::computeSampleBasisFunctions(samples.begin(), samples.end(), buildSettings.basisType);
        }

        KDNode &root = kdTree.getRoot();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        evaluateRegionsNode(kdTree, root, 1, samples, sampleRange, dataStorage, candidateDataStorage, buildSettings, field);

        // Postprocessing: decay signature when a region has way too many samples
        embree::parallel_for(dataStorage.size(), [&](embree::range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (dataStorage[i].first.candidate.signatures.getNumSamples() > PGL_SIGNATURE_MAX_SAMPLES)
                    dataStorage[i].first.candidate.signatures.decay(0.5f);
            }
        });
        embree::parallel_for(candidateDataStorage.size(), [&](embree::range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (candidateDataStorage[i].signatures.getNumSamples() > PGL_SIGNATURE_MAX_SAMPLES)
                    candidateDataStorage[i].signatures.decay(0.5f);
            }
        });
    }

    void prepareSampleReprojection(typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd, const SampleStatistics &stats, const Settings &settings) const {
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

        // if (settings.optimizeSignature)
        //     Signature::computeSampleBasisFunctions(samplesBegin, samplesEnd, settings.basisType);
    }

    void updateTreeNode(KDTree &kdTree, KDNode &node, uint8_t depth, uint8_t prevSplitDim, const BBox &bounds,
                        TSamplesContainer &samples, const Range &sampleRange, TZeroValueSamplesContainer &zeroSamples, const Range &zeroSampleRange,
                        tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage,
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
            SubdivisionData &candidate = region.candidate;
            // Avoid double counting when this node is a first-level split
            SampleStatistics mergedStats = candidate.sampleStatistics;
            if (region.candidate.updated) {   // a result of a recent signature-based split
                region.regionBounds = bounds;
            } else {
                mergedStats.merge(computeStats(samplesBegin, samplesEnd));
            }

            KDNode *nodeLR[2] = {nullptr, nullptr};
            bool triggersSplit = false;
            float defensiveThreshold = getDefensiveSampleCount(settings.forcedSampleCountThreshold, iteration, settings);

            if (depth + 1 <= settings.maxDepth && 
                (iteration < settings.initializingIters && mergedStats.getNumSamples() > settings.sampleCountThreshold) ||
                (iteration >= settings.initializingIters && mergedStats.getNumSamples() > defensiveThreshold)) {
                // 1. sample count threshold
                triggersSplit = true;
                bool hasCandidateSplit = iteration >= settings.initializingIters && candidate.hasSplit();
                uint32_t lChildIdx = candidate.lChildIdx;
                if (hasCandidateSplit) {
                    splitDim = candidate.dim;
                    splitPos = candidate.pivot;
                } else proposeSplit(samplesBegin, samplesEnd, mergedStats, splitDim, splitPos, settings);
                OPENPGL_ASSERT(splitDim < 3);

                auto rDataItr = dataStorage.emplace_back(region, Range());
                RegionType *regionLR[2] = {&region, &rDataItr->first};

                // Inheritance
                for (uint8_t c: {0, 1}) {
                    if (hasCandidateSplit) {
                        regionLR[c]->candidate = candidateDataStorage[lChildIdx + c];
                        regionLR[c]->candidate.energy = 0;
                        OPENPGL_ASSERT(regionLR[c]->candidate.depth == depth + 1);
                    } else {
                        regionLR[c]->candidate.sampleStatistics.split(splitDim, splitPos, settings.decayRatio, c);
                        regionLR[c]->candidate.depth = depth + 1;
                    }
                    if (iteration >= settings.initializingIters) {
                        clearCandidateSignatures(regionLR[c]->candidate, candidateDataStorage);
                    }
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
                // 2. Signature splitting
                Timer timer;
                
                std::vector<std::pair<uint32_t, uint32_t>> newLeafs;
                newLeafs.reserve(8);

                if (!candidate.updated || !settings.nonRecursive) {  // Update signatures at this parent node
                    if (settings.reproject)
                        prepareSampleReprojection(samplesBegin, samplesEnd, mergedStats, settings);
                    if (settings.singlePromotion) {
                        // Promote at most one level
                        if (updateCandidateRegionsOnePromotion(depth, kdTree, candidate, candidate, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd, candidateDataStorage, settings)) {
                            OPENPGL_ASSERT(candidate.hasSplit());
                            splitDim = candidate.dim, splitPos = candidate.pivot;
                            triggersSplit = true;

                            auto rDataItr = dataStorage.emplace_back(region, Range());
                            RegionType *regionLR[2] = {&region, &rDataItr->first};
                            uint32_t lChildIdx = candidate.lChildIdx;

                            // Inheritance
                            for (uint8_t c: {0, 1}) {
                                regionLR[c]->candidate = candidateDataStorage[lChildIdx + c];
                                regionLR[c]->candidate.energy = 0;
                                OPENPGL_ASSERT(regionLR[c]->candidate.depth == depth + 1);
                                clearCandidateSignatures(regionLR[c]->candidate, candidateDataStorage);
                                regionLR[c]->splitFlag += 1;
                                // (c ? regionLR[c]->regionBounds.lower[splitDim] : regionLR[c]->regionBounds.upper[splitDim]) = splitPos;
                                // regionBounds set later
                                OPENPGL_ASSERT(regionLR[c]->candidate.depth > depth);
                            }

                            // Extend KD tree
                            uint32_t nodeIdLeft = kdTree.addChildrenPair();
                            nodeLR[0] = &kdTree.getNode(nodeIdLeft);
                            nodeLR[1] = &kdTree.getNode(nodeIdLeft + 1);
                            node.setToInnerNode(splitDim, splitPos, nodeIdLeft);
                            nodeLR[0]->setDataNodeIdx(dataIdx);
                            nodeLR[1]->setDataNodeIdx(std::distance(dataStorage.begin(), rDataItr));
                        }
                    } else {
                        uint32_t leftNodeId = updateCandidateRegions(depth, kdTree, candidate, candidate, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd, candidateDataStorage, settings, newLeafs);
                        if (leftNodeId > 0) {  // has promotion
                            OPENPGL_ASSERT(candidate.hasSplit());
                            splitDim = candidate.dim, splitPos = candidate.pivot;
                            triggersSplit = true;
                            // Allocate data for the new leafs
                            std::vector<uint32_t> dataInds(newLeafs.size());
                            dataInds[0] = dataIdx;  // reuse
                            auto firstdataItr = dataStorage.grow_by(newLeafs.size() - 1, {region, Range()});
                            for (int i = 1; i < newLeafs.size(); ++i) {
                                dataInds[i] = std::distance(dataStorage.begin(), firstdataItr) + i - 1;
                            }

                            // Inheritance
                            for (int i = 0; i < newLeafs.size(); ++i) {
                                uint32_t newNodeId = newLeafs[i].first, canDataIdx = newLeafs[i].second;
                                KDNode &newNode = kdTree.getNode(newNodeId);
                                newNode.setDataNodeIdx(dataInds[i]);
                                RegionType &newRegion = dataStorage[dataInds[i]].first;
                                newRegion.candidate = candidateDataStorage[canDataIdx];
                                newRegion.candidate.energy = 0;
                                clearCandidateSignatures(newRegion.candidate, candidateDataStorage);
                                // regionBounds set later
                                OPENPGL_ASSERT(newRegion.candidate.depth > depth);
                                newRegion.splitFlag += newRegion.candidate.depth - depth;
                            }

                            // Extend KD tree
                            node.setToInnerNode(splitDim, splitPos, leftNodeId);
                            nodeLR[0] = &kdTree.getNode(leftNodeId);
                            nodeLR[1] = &kdTree.getNode(leftNodeId + 1);
                        }
                    }
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
                    [&] { updateTreeNode(kdTree, *nodeLR[0], depth + 1, splitDim, boundsLR.first, samples, sampleRangeLR[0], zeroSamples, zeroSampleRangeLR[0], dataStorage, candidateDataStorage, settings, iteration); },
                    [&] { updateTreeNode(kdTree, *nodeLR[1], depth + 1, splitDim, boundsLR.second, samples, sampleRangeLR[1], zeroSamples, zeroSampleRangeLR[1], dataStorage, candidateDataStorage, settings, iteration); }
                );
            } else {
                // No split! Just merge in new samples
                if (!region.candidate.updated) {
                    region.candidate.sampleStatistics = mergedStats;
                    region.candidate.sampleStatistics.addNumZeroValueSamples(zeroSampleRange.size());
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
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[0]), depth + 1, splitDim, boundsLR.first, samples, sampleRangesLR[0], zeroSamples, zeroSampleRangesLR[0], dataStorage, candidateDataStorage, settings, iteration); },
                [&]{ updateTreeNode(kdTree, kdTree.getNode(nodeIdsLR[1]), depth + 1, splitDim, boundsLR.second, samples, sampleRangesLR[1], zeroSamples, zeroSampleRangesLR[1], dataStorage, candidateDataStorage, settings, iteration); }
            );
        }
    }

    bool updateCandidateRegionsOnePromotion(uint8_t depth, KDTree &kdTree, const SubdivisionData &root, SubdivisionData &current,
        typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
        typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd,
        tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &settings) const {
        OPENPGL_ASSERT(depth == current.depth);
        OPENPGL_ASSERT(root.depth <= depth && depth <= settings.maxDepth);
        const uint8_t lookaheadLevel = current.depth - root.depth;
        OPENPGL_ASSERT(lookaheadLevel <= settings.lookaheadDepth);

        auto update = [&settings, &root](SubdivisionData &region,
            typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
            typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd) {
            if (!region.updated) {
                region.sampleStatistics.merge(computeStats(samplesBegin, samplesEnd));
                region.updated = true;
            } else {
                OPENPGL_ASSERT(region.signatures.getNumSamples() == 0);
            }
            region.signatures.addSamples(samplesBegin, samplesEnd, settings.signatureEnsembleConfig, settings.multiplyCosine);
            region.signatures.addZeroSamples(std::distance(zeroSamplesBegin, zeroSamplesEnd));
            region.energy = getDistance(region.signatures, root.signatures, settings);  // the root could change, so we need to recompute the distance even if updated
            region.risk = region.signatures.getRisk();
            switch (settings.confidenceType) {
                // case PGL_SPATIAL_CONFIDENCE_RISK: region.risk = region.signature.getRisk(); break;
                case PGL_SPATIAL_CONFIDENCE_TTEST: {
                    float mu = 0.5f * (region.signatures[0].getMean(0) + root.signatures[0].getMean(0));
                    float eps = (mu * settings.signatureDistanceThreshold) / (settings.tEpsK * settings.tValueThreshold);
                    // When k == 1, the converged T value (mu_1 - mu_2) / eps will equal to k * T when triggering the energy threshold
                    region.tValue = Signature::getWelchT(region.signatures[0], root.signatures[0], eps);
                    break;
                }
                default: break;
            }
        };

        // Update current
        if (lookaheadLevel == 0) {  // at the root
            // Filter outliers once at the root level
            samplesEnd = filterSamples(current, samplesBegin, samplesEnd, settings);
            update(current, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd);
            OPENPGL_ASSERT(current.energy == 0);
        }

        // Lookahead
        if (!current.hasSplit()
            && lookaheadLevel + 1 <= settings.lookaheadDepth
            && depth + 1 <= settings.maxDepth && current.sampleStatistics.getNumSamples() >= settings.minSamplesCandidateSplit) {  // propose a new split
            float splitPos;
            uint8_t splitDim;
            proposeSplit(samplesBegin, samplesEnd, current.sampleStatistics, splitDim, splitPos, settings);
            current.dim = splitDim, current.pivot = splitPos;
            current.lChildIdx = std::distance(candidateDataStorage.begin(), candidateDataStorage.grow_by(2));
            for (uint8_t c: {0, 1}) {
                SubdivisionData &child = candidateDataStorage[current.lChildIdx + c];
                child.sampleStatistics = current.sampleStatistics;
                child.sampleStatistics.split(splitDim, splitPos, settings.decayRatio, c);
                child.depth = depth + 1;
            }
        }

        if (current.hasSplit()) {
            // Split samples
            auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, current.dim, current.pivot);
            auto zeroSamplesMid = pivotSplitSamples(zeroSamplesBegin, zeroSamplesEnd, current.dim, current.pivot);

            SubdivisionData &left = candidateDataStorage[current.lChildIdx];
            SubdivisionData &right = candidateDataStorage[current.lChildIdx + 1];

            // Update L/R signatures
            update(left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid);
            update(right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd);

            // float energyLR = Signature::getDistance(left.signature, right.signature, settings.stdMultiplier);

            // Try promotion of the current split: either child should exceed the energy threshold
            if (checkPromotion(root, left, right, settings))
                return true;

            if (lookaheadLevel + 1 < settings.lookaheadDepth) {
                // Update L/R recursively
                return updateCandidateRegionsOnePromotion(depth + 1, kdTree, root, left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid, candidateDataStorage, settings) ||
                       updateCandidateRegionsOnePromotion(depth + 1, kdTree, root, right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd, candidateDataStorage, settings);
            }
        }

        return false;
    }

    // Update and try promotion recursively at the candidate nodes beneath current
    // Constructs the subtree and returns the *kdNode idx of the left child to current* if there is any promotion under current
    // Outputs new leaf nodes' (kd node id, candidate data idx) into newLeafs, in the DFS order
    uint32_t updateCandidateRegions(uint8_t depth, KDTree &kdTree, const SubdivisionData &root, SubdivisionData &current,
        typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
        typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd,
        tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &settings, std::vector<std::pair<uint32_t, uint32_t>> &newLeafs) const {
        OPENPGL_ASSERT(depth == current.depth);
        OPENPGL_ASSERT(root.depth <= depth && depth <= settings.maxDepth);
        const uint8_t lookaheadLevel = current.depth - root.depth;
        OPENPGL_ASSERT(lookaheadLevel <= settings.lookaheadDepth);

        auto update = [&settings, &root](SubdivisionData &region,
            typename TSamplesContainer::iterator samplesBegin, typename TSamplesContainer::iterator samplesEnd,
            typename TZeroValueSamplesContainer::iterator zeroSamplesBegin, typename TZeroValueSamplesContainer::iterator zeroSamplesEnd) {
            if (!region.updated) {
                region.sampleStatistics.merge(computeStats(samplesBegin, samplesEnd));
                region.updated = true;
            } else {
                OPENPGL_ASSERT(region.signatures.getNumSamples() == 0);
            }
            region.signatures.addSamples(samplesBegin, samplesEnd, settings.signatureEnsembleConfig, settings.multiplyCosine);
            region.signatures.addZeroSamples(std::distance(zeroSamplesBegin, zeroSamplesEnd));
            region.energy = getDistance(region.signatures, root.signatures, settings);  // the root could change, so we need to recompute the distance even if updated
            region.risk = region.signatures.getRisk();
            switch (settings.confidenceType) {
                // case PGL_SPATIAL_CONFIDENCE_RISK: region.risk = region.signature.getRisk(); break;
                case PGL_SPATIAL_CONFIDENCE_TTEST: {
                    float mu = 0.5f * (region.signatures[0].getMean(0) + root.signatures[0].getMean(0));
                    float eps = (mu * settings.signatureDistanceThreshold) / (settings.tEpsK * settings.tValueThreshold);
                    // When k == 1, the converged T value (mu_1 - mu_2) / eps will equal to k * T when triggering the energy threshold
                    region.tValue = Signature::getWelchT(region.signatures[0], root.signatures[0], eps);
                    break;
                }
                default: break;
            }
        };

        // Update current
        if (lookaheadLevel == 0) {  // at the root
            // Filter outliers once at the root level
            samplesEnd = filterSamples(current, samplesBegin, samplesEnd, settings);
            update(current, samplesBegin, samplesEnd, zeroSamplesBegin, zeroSamplesEnd);
            OPENPGL_ASSERT(current.energy == 0);
        }

        // Lookahead
        if (!current.hasSplit()
            && lookaheadLevel + 1 <= settings.lookaheadDepth
            && depth + 1 <= settings.maxDepth && current.sampleStatistics.getNumSamples() >= settings.minSamplesCandidateSplit) {  // propose a new split
            float splitPos;
            uint8_t splitDim;
            proposeSplit(samplesBegin, samplesEnd, current.sampleStatistics, splitDim, splitPos, settings);
            current.dim = splitDim, current.pivot = splitPos;
            current.lChildIdx = std::distance(candidateDataStorage.begin(), candidateDataStorage.grow_by(2));
            for (uint8_t c: {0, 1}) {
                SubdivisionData &child = candidateDataStorage[current.lChildIdx + c];
                child.sampleStatistics = current.sampleStatistics;
                child.sampleStatistics.split(splitDim, splitPos, settings.decayRatio, c);
                child.depth = depth + 1;
            }
        }

        if (current.hasSplit()) {
            // Split samples
            auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, current.dim, current.pivot);
            auto zeroSamplesMid = pivotSplitSamples(zeroSamplesBegin, zeroSamplesEnd, current.dim, current.pivot);

            SubdivisionData &left = candidateDataStorage[current.lChildIdx];
            SubdivisionData &right = candidateDataStorage[current.lChildIdx + 1];

            // Update L/R signatures
            update(left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid);
            update(right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd);

            // float energyLR = getDistance(left.signature, right.signature, settings.stdMultiplier);

            // Try promotion of the current split: either child should exceed the energy threshold
            bool promoteCurrentSplit = checkPromotion(root, left, right, settings);

            uint32_t leftLeftNodeId = 0, rightLeftNodeId = 0;
            if (!promoteCurrentSplit && lookaheadLevel + 1 < settings.lookaheadDepth) {
                // Update L/R recursively
                leftLeftNodeId = updateCandidateRegions(depth + 1, kdTree, root, left, samplesBegin, samplesMid, zeroSamplesBegin, zeroSamplesMid, candidateDataStorage, settings, newLeafs);
                rightLeftNodeId = updateCandidateRegions(depth + 1, kdTree, root, right, samplesMid, samplesEnd, zeroSamplesMid, zeroSamplesEnd, candidateDataStorage, settings, newLeafs);
            }

            // Extend KD tree if: 1) current split is promoted, 2) left or right child has promotion
            if (promoteCurrentSplit || leftLeftNodeId || rightLeftNodeId) {
                uint32_t leftNodeId = kdTree.addChildrenPair();
                KDNode &leftNode = kdTree.getNode(leftNodeId), &rightNode = kdTree.getNode(leftNodeId + 1);
                if (leftLeftNodeId) {
                    OPENPGL_ASSERT(left.hasSplit());
                    leftNode.setToInnerNode(left.dim, left.pivot, leftLeftNodeId);
                }
                else {
                    newLeafs.emplace_back(leftNodeId, (uint32_t) current.lChildIdx);
                    leftNode.setLeaf();  // The updateTreeNode method will fill in the missing data indices
                }
                if (rightLeftNodeId) {
                    OPENPGL_ASSERT(right.hasSplit());
                    rightNode.setToInnerNode(right.dim, right.pivot, rightLeftNodeId);
                }
                else {
                    newLeafs.emplace_back(leftNodeId + 1, (uint32_t) current.lChildIdx + 1);
                    rightNode.setLeaf();
                }
                return leftNodeId;
            }
        }

        return 0;
    }

    static float getDistance(const SignatureEnsemble &a, const SignatureEnsemble &b, const Settings &settings) {
        if (settings.confidenceType == PGL_SPATIAL_CONFIDENCE_TTEST_PER_BIN)
            return Signature::getDistanceTTest(a[0], b[0], settings.stdMultiplier, settings.tValueThreshold);
        else
            return SignatureEnsemble::getDistance(a, b, settings.stdMultiplier);
    }

    static bool checkPromotion(const SubdivisionData &root, const SubdivisionData &left, const SubdivisionData &right, const Settings &settings) {
        if (!settings.enablePromotion) return false;
        switch (settings.confidenceType) {
            case PGL_SPATIAL_CONFIDENCE_NONE:
            case PGL_SPATIAL_CONFIDENCE_TTEST_PER_BIN:
                return left.signatures.getNumSamples() > settings.minSamplesPromotion && right.signatures.getNumSamples() > settings.minSamplesPromotion &&
                       (
                           // (energyLR > settings.signatureDistanceThreshold) ||  // LR
                           left.energy > settings.signatureDistanceThreshold || // P and L
                           right.energy > settings.signatureDistanceThreshold // P and R
                       );
            case PGL_SPATIAL_CONFIDENCE_RISK:
                return root.risk <= settings.riskTolerance &&
                       left.signatures.getNumSamples() > settings.minSamplesPromotion && right.signatures.getNumSamples() > settings.minSamplesPromotion &&
                       (
                           // (left.risk <= settings.riskTolerance && right.risk <= settings.riskTolerance && energyLR > settings.signatureDistanceThreshold) || // LR
                           (left.risk <= settings.riskTolerance && left.energy > settings.signatureDistanceThreshold) ||  // P and L
                           (right.risk <= settings.riskTolerance && right.energy > settings.signatureDistanceThreshold)   // P and R
                       );
            case PGL_SPATIAL_CONFIDENCE_TTEST:
                return left.signatures.getNumSamples() > settings.minSamplesPromotion && right.signatures.getNumSamples() > settings.minSamplesPromotion &&
                       (
                           // (left.risk <= settings.riskTolerance && right.risk <= settings.riskTolerance && energyLR > settings.signatureDistanceThreshold) || // LR
                           (std::abs(left.tValue) > settings.tValueThreshold /*&& left.energy > settings.signatureDistanceThreshold*/) || // P and L
                           (std::abs(right.tValue) > settings.tValueThreshold /*&& right.energy > settings.signatureDistanceThreshold*/)  // P and R
                       );
            default:
                std::cerr << "Unknown confidence type" << std::endl;
                return false;
        }
    }

    static typename TSamplesContainer::iterator filterSamples(SubdivisionData &data, typename TSamplesContainer::iterator begin, typename TSamplesContainer::iterator end, const Settings &settings) {
        switch (settings.filterType) {
            case PGL_SPATIAL_FILTER_NONE: return end;
            case PGL_SPATIAL_FILTER_PERCENTAGE: {
                // Partition the samples according to their weight, and return the pivot pointing at 100 inlierPercent %
                const size_t N = std::distance(begin, end);
                auto pivot = begin + std::min((size_t) (settings.inlierPercent * N), N);
                std::nth_element(begin, pivot, end, [](auto a, auto b) { return a.weight < b.weight; });
                // std::sort(begin, end, [](auto a, auto b) { return a.weight < b.weight; });
                return pivot;
            }
            case PGL_SPATIAL_FILTER_DBOR: {
                // First compute the mean and std
                double mean, m2;
                computeWeightMoments(begin, end, mean, m2);
                double sigma = std::sqrt(m2 / (double) std::distance(begin, end));
                // Filter the samples at pivot mean + k * std
                return std::partition(begin, end, [=](auto s) { return s.weight <= mean + settings.DBORstdMultiplier * sigma; });
            }
            case PGL_SPATIAL_FILTER_DBOR_ACCUM: {
                // First compute the mean and std
                if (!data.updated || data.sampleStatistics.weightCnt == 0)
                    accumulateWeightMoments(begin, end, data.sampleStatistics.weightMean, data.sampleStatistics.weightM2, data.sampleStatistics.weightCnt);
                // else: data.weightM2 must have been updated with the current samples
                float sigma = data.sampleStatistics.getDBORStd();
                // Filter the samples at pivot mean + k * std
                return std::partition(begin, end, [=](auto s) { return s.weight <= data.sampleStatistics.weightMean + settings.DBORstdMultiplier * sigma; });
            }
            default: throw std::runtime_error("Unknown filter type");
        }
    }

    static float getDefensiveSampleCount(uint32_t c0, uint32_t iteration, const Settings &settings) {
        if (c0 == (uint32_t) -1) return std::numeric_limits<float>::infinity();
        iteration++;  // 1, 2, 3, ..
        switch (settings.defensiveType) {
            case PGL_SPATIAL_DEFENSIVE_FIXED: return c0;
            case PGL_SPATIAL_DEFENSIVE_SQRT: return c0 * std::sqrt(iteration);
            case PGL_SPATIAL_DEFENSIVE_PPG: {
                // In alignment with PPG
                // * sqrt(2) from c0 when iteration = 2^k {at iter 1, 2, 4, 8, ...}
                float c = c0;
                constexpr float scale = std::sqrt(2.f);
                while (iteration > 1) {
                    iteration >>= 1;
                    c *= scale;
                }
                return c;
            }
            default: throw std::runtime_error("Unknown defensive type");
        };
    }

    // Clear the signatures beneath current
    void clearCandidateSignatures(SubdivisionData &current, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage) const {
        current.signatures.clear();
        current.energy = current.risk = current.tValue = 0;
        if (current.hasSplit()) {
            SubdivisionData &left = candidateDataStorage[current.lChildIdx];
            SubdivisionData &right = candidateDataStorage[current.lChildIdx + 1];
            clearCandidateSignatures(left, candidateDataStorage);
            clearCandidateSignatures(right, candidateDataStorage);
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
    void evaluateRegionsNode(KDTree &kdTree, KDNode &node, uint8_t depth, TContainer &samples, const Range sampleRange, tbb::concurrent_vector<std::pair<TRegion, Range> > &dataStorage, tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &buildSettings, const FieldType &field) const
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
        tbb::concurrent_vector<SubdivisionData> &candidateDataStorage, const Settings &settings) const {
        constexpr bool isNonZeroSample = has_member_weight<typename TContainer::value_type>::value;

        // Update self
        if constexpr (isNonZeroSample) {
            current.signatures.addSamples(samplesBegin, samplesEnd, settings.signatureEnsembleConfig, settings.multiplyCosine);
        } else {
            current.signatures.addZeroSamples(std::distance(samplesBegin, samplesEnd));
        }
        current.energy = getDistance(current.signatures, root.signatures, settings);
        current.risk = current.signatures.getRisk();
        switch (settings.confidenceType) {
            // case PGL_SPATIAL_CONFIDENCE_RISK: current.risk = current.signature.getRisk(); break;
            case PGL_SPATIAL_CONFIDENCE_TTEST: {
                float mu = 0.5f * (current.signatures[0].getMean(0) + root.signatures[0].getMean(0));
                float eps = (mu * settings.signatureDistanceThreshold) / (settings.tEpsK * settings.tValueThreshold);
                // When k == 1, the converged T value (mu_1 - mu_2) / eps will equal to k * T when triggering the energy threshold
                current.tValue = Signature::getWelchT(current.signatures[0], root.signatures[0], eps);
                break;
            }
            default: break;
        }

        if (current.hasSplit()) {
            // Split samples
            auto samplesMid = pivotSplitSamples(samplesBegin, samplesEnd, current.dim, current.pivot);

            // Update L/R
            SubdivisionData &leftRegion = candidateDataStorage[current.lChildIdx];
            SubdivisionData &rightRegion = candidateDataStorage[current.lChildIdx + 1];

            evaluateCandidateRegions<TContainer>(kdTree, root, leftRegion, samplesBegin, samplesMid, candidateDataStorage, settings);
            evaluateCandidateRegions<TContainer>(kdTree, root, rightRegion, samplesMid, samplesEnd, candidateDataStorage, settings);
        }
    }

    template<typename SampleIterator>
    inline float proposeSplit(SampleIterator begin, SampleIterator end, const SampleStatistics &stats, uint8_t &splitDim, float &splitPos, const Settings &buildSettings) const {
        size_t minSamplesPerSide = buildSettings.minSamplesCandidateSplit / 2;
        if (std::distance(begin, end) <= buildSettings.minSamplesCandidateSplit) {
            splitBaseline(stats, splitDim, splitPos);
            return 0;
        }
        switch (buildSettings.splitType) {
            case PGL_SPATIAL_SPLIT_BASELINE: splitBaseline(stats, splitDim, splitPos); return 0;

            case PGL_SPATIAL_SPLIT_VS: return varianceScan(begin, end, stats, minSamplesPerSide, 0, splitDim, splitPos, buildSettings);
            case PGL_SPATIAL_SPLIT_IGS: return informationGainScan(begin, end, stats, minSamplesPerSide, 1.0, splitDim, splitPos, buildSettings);
            case PGL_SPATIAL_SPLIT_FS: return fluenceScan(begin, end, stats, minSamplesPerSide, 0.03, splitDim, splitPos, buildSettings);

            default: throw std::runtime_error("Unknown split type");
        }
    }

    // Compute the candidate split position, return the gain estimate that will guide when to split
    // template<typename SampleIterator>
    // inline float proposeSplit(uint8_t prevSplitDim, const BBox &bounds, SampleIterator begin, SampleIterator end, const SampleStatistics &stats,
    //                           uint8_t &splitDim, float &splitPos, const Settings &buildSettings) const {
    //     size_t minSamplesPerSide = buildSettings.minSamplesCandidateSplit / 2;
    //     switch (buildSettings.splitType) {
    //         case PGL_SPATIAL_SPLIT_BASELINE: splitBaseline(stats, splitDim, splitPos); return 0;
    //         case PGL_SPATIAL_SPLIT_ROUNDROBIN: splitRoundRobin(prevSplitDim, stats, splitDim, splitPos); return 0;
    //         case PGL_SPATIAL_SPLIT_PPG: splitPPG(prevSplitDim, bounds, stats, splitDim, splitPos); return 0;
    //
    //         case PGL_SPATIAL_SPLIT_VS: return varianceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
    //         case PGL_SPATIAL_SPLIT_COVS: return covarianceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
    //         case PGL_SPATIAL_SPLIT_IGS: return informationGainScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
    //         case PGL_SPATIAL_SPLIT_FS: return fluenceScan(begin, end, stats, minSamplesPerSide, buildSettings.defensiveness, splitDim, splitPos);
    //
    //         default: throw std::runtime_error("Unknown split type");
    //     }
    // }

    void splitBaseline(const SampleStatistics &stats, uint8_t &splitDim, float &splitPos) const
    {
        splitDim = maxDimension(stats.getVariance());
        splitPos = stats.getMean()[splitDim];
    }

    void splitRoundRobin(uint8_t prevSplitDim, const SampleStatistics &stats, uint8_t &splitDim, float &splitPos, const Settings &settings) const
    {
        Vector3 var = stats.getVariance();
        float maxVar = reduce_max(var);
        splitDim = (prevSplitDim + 1) % 3;
        while (var[splitDim] < settings.varianceThreshold * maxVar) {
            splitDim = (splitDim + 1) % 3;  // skip dimensions with low variance
        }
        splitPos = stats.getMean()[splitDim];
    }

    void splitPPG(uint8_t prevSplitDim, const BBox &bounds, const SampleStatistics &stats, uint8_t &splitDim, float &splitPos, const Settings &settings) const
    {
        Vector3 var = stats.getVariance();
        float maxVar = reduce_max(var);
        splitDim = (prevSplitDim + 1) % 3;
        while (var[splitDim] < settings.varianceThreshold * maxVar) {
            splitDim = (splitDim + 1) % 3;  // skip dimensions with low variance
        }
        splitPos = bounds.center()[splitDim];
    }

    template <typename SampleIterator>
    float varianceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos, const Settings &settings) const
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
            if (posVariances[dim] < settings.varianceThreshold * maxPosVariance) {
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
    float covarianceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos, const Settings &settings) const
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
            if (posVariances[dim] < settings.varianceThreshold * maxPosVariance) continue;  // skip dimensions with low variance

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
    float informationGainScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos, const Settings &settings) const
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
            if (posVariances[dim] < settings.varianceThreshold * maxPosVariance) continue;  // skip dimensions with low variance

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
        // Pulling split slightly towards the center
        const float delta = std::sqrt(maxPosVariance * settings.varianceThreshold);
        const float meanPos = sampleStats.getMean()[splitDim];
        if (splitPos + delta < meanPos) {
            splitPos += delta;
        } else if (splitPos - delta > meanPos) {
            splitPos -= delta;
        }
        return (float) gains[splitDim];  // use the absolute gain
    }

    template <typename SampleIterator>
    float fluenceScan(const SampleIterator begin, const SampleIterator end, const SampleStatistics &sampleStats, const size_t minSamplesPerSide, const float defensiveness, uint8_t &splitDim, float &splitPos, const Settings &settings) const
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
            if (posVariances[dim] < settings.varianceThreshold * maxPosVariance) {
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

    template <typename SampleIterator>
    static void accumulateWeightMoments(const SampleIterator begin, const SampleIterator end, float &mean, float &M2, float &cnt) {
        for (auto it = begin; it != end; ++it) {
            cnt++;
            float w = it->weight;
            float delta = w - mean;
            mean += delta / cnt;
            float delta2 = w - mean;
            M2 += delta * delta2;
        }
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
    ss << "  maxDepth: " << maxDepth << std::endl;
    ss << "  minSamplesCandidateSplit: " << minSamplesCandidateSplit << std::endl;
    ss << "  minSamplesPromotion: " << minSamplesPromotion << std::endl;
    ss << "  sampleCountThreshold: " << sampleCountThreshold << std::endl;
    ss << "  forcedSampleCountThreshold: " << forcedSampleCountThreshold << std::endl;
    ss << "  initializingIters: " << initializingIters << std::endl;
    ss << "  lookaheadDepth: " << lookaheadDepth << std::endl;
    ss << "  signatureDistanceThreshold: " << signatureDistanceThreshold << std::endl;
    ss << "  decayRatio: " << decayRatio << std::endl;
    ss << "  defensiveness: " << defensiveness << std::endl;
    ss << "  stdMultiplier: " << stdMultiplier << std::endl;
    ss << "  riskTolerance: " << riskTolerance << std::endl;
    ss << "  tValueThreshold: " << tValueThreshold << std::endl;
    ss << "  inlierPercent: " << inlierPercent << std::endl;
    ss << "  DBORstdMultiplier: " << DBORstdMultiplier << std::endl;
    ss << "  tEpsK: " << tEpsK << std::endl;
    ss << "  varianceThreshold: " << varianceThreshold << std::endl;
    ss << "  enablePromotion: " << enablePromotion << std::endl;
    ss << "  multiplyCosine: " << multiplyCosine << std::endl;
    ss << "  reproject: " << reproject << std::endl;
    ss << "  nonRecursive: " << nonRecursive << std::endl;
    ss << "  singlePromotion: " << singlePromotion << std::endl;
    ss << "  optimizeSignature: " << optimizeSignature << std::endl;
    ss << "  confidenceType: " << confidenceType << std::endl;
    ss << "  defensiveType: " << defensiveType << std::endl;
    ss << "  filterType: " << filterType << std::endl;
    ss << "  # signatures: " << signatureEnsembleConfig.size() << std::endl;
    for (size_t i = 0; i < signatureEnsembleConfig.size(); ++i) {
        auto &cfg = signatureEnsembleConfig[i];
        ss << "    " << i << ": " << cfg.numBins << " bins";
        switch (cfg.basisType) {
            case PGL_BASIS_FUNC_NN:
            case PGL_BASIS_FUNC_LATITUDE:
            case PGL_BASIS_FUNC_LONGITUDE:
                ss << ", res = " << cfg.getResolution();
                break;
            case PGL_BASIS_FUNC_SPLAT:
                ss << ", res = " << cfg.getResolution() << ", sigma = " << cfg.getSplatSigma();
                break;
            case PGL_BASIS_FUNC_DON_PCG:
            case PGL_BASIS_FUNC_DON_XI:
                ss << ", " << cfg.getOctaveMin() << ".." << cfg.getOctaveMax() << " octaves, gamma = " << cfg.getDONGamma();
                break;
        }
        ss << std::endl;
    }

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
    stream.write(reinterpret_cast<const char*>(&forcedSampleCountThreshold), sizeof(forcedSampleCountThreshold));
    stream.write(reinterpret_cast<const char*>(&initializingIters), sizeof(initializingIters));
    stream.write(reinterpret_cast<const char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.write(reinterpret_cast<const char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.write(reinterpret_cast<const char*>(&decayRatio), sizeof(decayRatio));
    stream.write(reinterpret_cast<const char*>(&defensiveness), sizeof(defensiveness));
    stream.write(reinterpret_cast<const char*>(&enablePromotion), sizeof(enablePromotion));
    stream.write(reinterpret_cast<const char*>(&stdMultiplier), sizeof(stdMultiplier));
    stream.write(reinterpret_cast<const char*>(&riskTolerance), sizeof(riskTolerance));
    stream.write(reinterpret_cast<const char*>(&tValueThreshold), sizeof(tValueThreshold));
    stream.write(reinterpret_cast<const char*>(&inlierPercent), sizeof(inlierPercent));
    stream.write(reinterpret_cast<const char*>(&DBORstdMultiplier), sizeof(DBORstdMultiplier));
    stream.write(reinterpret_cast<const char*>(&tEpsK), sizeof(tEpsK));
    stream.write(reinterpret_cast<const char*>(&varianceThreshold), sizeof(varianceThreshold));
    stream.write(reinterpret_cast<const char*>(&multiplyCosine), sizeof(multiplyCosine));
    stream.write(reinterpret_cast<const char*>(&reproject), sizeof(reproject));
    stream.write(reinterpret_cast<const char*>(&nonRecursive), sizeof(nonRecursive));
    stream.write(reinterpret_cast<const char*>(&singlePromotion), sizeof(singlePromotion));
    stream.write(reinterpret_cast<const char*>(&optimizeSignature), sizeof(optimizeSignature));
    stream.write(reinterpret_cast<const char*>(&confidenceType), sizeof(confidenceType));
    stream.write(reinterpret_cast<const char*>(&defensiveType), sizeof(defensiveType));
    stream.write(reinterpret_cast<const char*>(&filterType), sizeof(filterType));
    uint32_t numSignatures = signatureEnsembleConfig.size();
    stream.write(reinterpret_cast<const char*>(&numSignatures), sizeof(uint32_t));
    for (const SignatureArguments &cfg: signatureEnsembleConfig) {
        stream.write(reinterpret_cast<const char*>(&cfg), sizeof(SignatureArguments));
    }
}

template<class TRegion, typename TSamplesContainer, typename TZeroValueSamplesContainer, typename TSamplingDistribution>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TZeroValueSamplesContainer, TSamplingDistribution>::Settings::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&splitType), sizeof(splitType));
    stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(maxDepth));
    stream.read(reinterpret_cast<char*>(&minSamplesCandidateSplit), sizeof(minSamplesCandidateSplit));
    stream.read(reinterpret_cast<char*>(&minSamplesPromotion), sizeof(minSamplesPromotion));
    stream.read(reinterpret_cast<char*>(&sampleCountThreshold), sizeof(sampleCountThreshold));
    stream.read(reinterpret_cast<char*>(&forcedSampleCountThreshold), sizeof(forcedSampleCountThreshold));
    stream.read(reinterpret_cast<char*>(&initializingIters), sizeof(initializingIters));
    stream.read(reinterpret_cast<char*>(&lookaheadDepth), sizeof(lookaheadDepth));
    stream.read(reinterpret_cast<char*>(&signatureDistanceThreshold), sizeof(signatureDistanceThreshold));
    stream.read(reinterpret_cast<char*>(&decayRatio), sizeof(decayRatio));
    stream.read(reinterpret_cast<char*>(&defensiveness), sizeof(defensiveness));
    stream.read(reinterpret_cast<char*>(&enablePromotion), sizeof(enablePromotion));
    stream.read(reinterpret_cast<char*>(&stdMultiplier), sizeof(stdMultiplier));
    stream.read(reinterpret_cast<char*>(&riskTolerance), sizeof(riskTolerance));
    stream.read(reinterpret_cast<char*>(&tValueThreshold), sizeof(tValueThreshold));
    stream.read(reinterpret_cast<char*>(&inlierPercent), sizeof(inlierPercent));
    stream.read(reinterpret_cast<char*>(&DBORstdMultiplier), sizeof(DBORstdMultiplier));
    stream.read(reinterpret_cast<char*>(&tEpsK), sizeof(tEpsK));
    stream.read(reinterpret_cast<char*>(&varianceThreshold), sizeof(varianceThreshold));
    stream.read(reinterpret_cast<char*>(&multiplyCosine), sizeof(multiplyCosine));
    stream.read(reinterpret_cast<char*>(&reproject), sizeof(reproject));
    stream.read(reinterpret_cast<char*>(&nonRecursive), sizeof(nonRecursive));
    stream.read(reinterpret_cast<char*>(&singlePromotion), sizeof(singlePromotion));
    stream.read(reinterpret_cast<char*>(&optimizeSignature), sizeof(optimizeSignature));
    stream.read(reinterpret_cast<char*>(&confidenceType), sizeof(confidenceType));
    stream.read(reinterpret_cast<char*>(&defensiveType), sizeof(defensiveType));
    stream.read(reinterpret_cast<char*>(&filterType), sizeof(filterType));
    uint32_t numSignatures = 0;
    stream.read(reinterpret_cast<char*>(&numSignatures), sizeof(uint32_t));
    signatureEnsembleConfig.resize(numSignatures);
    for (SignatureArguments &cfg: signatureEnsembleConfig) {
        stream.read(reinterpret_cast<char*>(&cfg), sizeof(SignatureArguments));
    }
}

}

#undef THRESHOLD_VAR_RATIO
