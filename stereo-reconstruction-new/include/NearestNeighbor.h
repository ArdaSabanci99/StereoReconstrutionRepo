#pragma once
#include <flann/flann.hpp>
#include "Eigen.h"

struct Match {
    int   idx;
    float weight;
};

class NearestNeighborSearch {
public:
    virtual ~NearestNeighborSearch() {}

    virtual void setMatchingMaxDistance(float maxDistance) {
        m_maxDistance = maxDistance;
    }

    virtual void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) = 0;
    virtual std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) = 0;

protected:
    float m_maxDistance;
    NearestNeighborSearch() : m_maxDistance{ 0.005f } {}
};


// Brute-force nearest neighbor search (reference / small clouds)
class NearestNeighborSearchBruteForce : public NearestNeighborSearch {
public:
    NearestNeighborSearchBruteForce() : NearestNeighborSearch() {}

    void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) override {
        m_points = targetPoints;
    }

    std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) override {
        const unsigned nMatches = transformedPoints.size();
        std::vector<Match> matches(nMatches);

        #pragma omp parallel for
        for (int i = 0; i < (int)nMatches; i++)
            matches[i] = getClosestPoint(transformedPoints[i]);

        return matches;
    }

private:
    std::vector<Eigen::Vector3f> m_points;

    Match getClosestPoint(const Vector3f& p) {
        int   idx = -1;
        float minDist = std::numeric_limits<float>::max();
        for (unsigned i = 0; i < m_points.size(); ++i) {
            float dist = (p - m_points[i]).norm();
            if (dist < minDist) { minDist = dist; idx = i; }
        }
        return (minDist <= m_maxDistance) ? Match{idx, 1.f} : Match{-1, 0.f};
    }
};


// FLANN kd-tree nearest neighbor search (fast, large clouds)
class NearestNeighborSearchFlann : public NearestNeighborSearch {
public:
    NearestNeighborSearchFlann() :
        NearestNeighborSearch(), m_nTrees{1}, m_index{nullptr}, m_flatPoints{nullptr} {}

    ~NearestNeighborSearchFlann() {
        if (m_index) {
            delete m_flatPoints;
            delete m_index;
            m_flatPoints = nullptr;
            m_index      = nullptr;
        }
    }

    void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) override {
        std::cout << "[FLANN] Building index with " << targetPoints.size() << " points.\n";

        m_flatPoints = new float[targetPoints.size() * 3];
        for (size_t i = 0; i < targetPoints.size(); i++)
            for (size_t d = 0; d < 3; d++)
                m_flatPoints[i * 3 + d] = targetPoints[i][d];

        flann::Matrix<float> dataset(m_flatPoints, targetPoints.size(), 3);
        m_index = new flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(m_nTrees));
        m_index->buildIndex();

        std::cout << "[FLANN] Index built.\n";
    }

    std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) override {
        if (!m_index) {
            std::cout << "[FLANN] Index must be built before querying.\n";
            return {};
        }

        float* queryData = new float[transformedPoints.size() * 3];
        for (size_t i = 0; i < transformedPoints.size(); i++)
            for (size_t d = 0; d < 3; d++)
                queryData[i * 3 + d] = transformedPoints[i][d];

        flann::Matrix<float> query(queryData, transformedPoints.size(), 3);
        flann::Matrix<int>   indices(new int[query.rows],   query.rows, 1);
        flann::Matrix<float> distances(new float[query.rows], query.rows, 1);

        flann::SearchParams searchParams{16};
        searchParams.cores = 0;
        m_index->knnSearch(query, indices, distances, 1, searchParams);

        std::vector<Match> matches;
        matches.reserve(transformedPoints.size());
        for (size_t i = 0; i < transformedPoints.size(); ++i) {
            if (*distances[i] <= m_maxDistance)
                matches.push_back(Match{*indices[i], 1.f});
            else
                matches.push_back(Match{-1, 0.f});
        }

        delete[] query.ptr();
        delete[] indices.ptr();
        delete[] distances.ptr();

        return matches;
    }

private:
    int   m_nTrees;
    flann::Index<flann::L2<float>>* m_index;
    float* m_flatPoints;
};
