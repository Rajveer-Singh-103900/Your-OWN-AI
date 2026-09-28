#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <array>
#include <filesystem>
#include <cstring>
#include <ctime>
#include <stdexcept>
#ifndef _WIN32
#include <sys/stat.h>
#endif

static const int DIMS = 16;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// =====================================================================
//  DATA TYPES
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot=0, na=0, nb=0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) s += std::abs(a[i]-b[i]);
    return s;
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }
};

// =====================================================================
//  KD-TREE
// =====================================================================

struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return; destroy(n->left); destroy(n->right); delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
        else                              n->right = ins(n->right, v, d+1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        float dn = dist(q, n->item.emb);
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop();
        }
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  VECTOR DATABASE  (demo 16D index)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
};

// =====================================================================
//  JSON HELPERS
// =====================================================================

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if ((unsigned char)c < 0x20) {   // other control chars are invalid raw in JSON
            char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned char)c); o += buf;
        }
        else                o += c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        try { v.push_back(std::stof(t)); } catch (...) {}
    return v;
}

// Minimal JSON reading for flat request bodies like {"title":"...","text":"...","k":3}.
// Keys are located by walking the top-level object, so a key name appearing
// inside some string value (e.g. a document containing the word "text") can't be mistaken for it.

static size_t jSkipStr(const std::string& b, size_t p) {       // p at opening quote
    for (p++; p < b.size(); p++) {
        if (b[p] == '\\') p++;
        else if (b[p] == '"') return p + 1;
    }
    return b.size();
}

static size_t jSkipVal(const std::string& b, size_t p) {
    if (p >= b.size()) return p;
    if (b[p] == '"') return jSkipStr(b, p);
    if (b[p] == '{' || b[p] == '[') {
        int depth = 0;
        while (p < b.size()) {
            char c = b[p];
            if (c == '"') { p = jSkipStr(b, p); continue; }
            if (c == '{' || c == '[') depth++;
            else if ((c == '}' || c == ']') && --depth == 0) return p + 1;
            p++;
        }
        return p;
    }
    while (p < b.size() && b[p] != ',' && b[p] != '}' && b[p] != ']') p++;
    return p;
}

// Index of the value belonging to a top-level key, or npos
static size_t jFindKey(const std::string& b, const std::string& key) {
    size_t p = b.find('{');
    if (p == std::string::npos) return p;
    p++;
    auto ws = [&]{ while (p < b.size() && std::isspace((unsigned char)b[p])) p++; };
    while (true) {
        ws();
        if (p >= b.size() || b[p] != '"') return std::string::npos;
        size_t ks = p;
        p = jSkipStr(b, p);
        bool match = b.compare(ks + 1, p - ks - 2, key) == 0 && p - ks - 2 == key.size();
        ws();
        if (p >= b.size() || b[p] != ':') return std::string::npos;
        p++; ws();
        if (match) return p;
        p = jSkipVal(b, p); ws();
        if (p < b.size() && b[p] == ',') { p++; continue; }
        return std::string::npos;
    }
}

static void appendUtf8(std::string& o, uint32_t cp) {
    if (cp < 0x80)         o += (char)cp;
    else if (cp < 0x800)   { o += (char)(0xC0 | (cp >> 6));  o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F));
                             o += (char)(0x80 | (cp & 0x3F)); }
    else                   { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F));
                             o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
}

static int hex4(const std::string& b, size_t p) {              // -1 if not 4 hex digits
    if (p + 4 > b.size()) return -1;
    int v = 0;
    for (size_t i = p; i < p + 4; i++) {
        char c = b[i]; v <<= 4;
        if      (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

// Extract a JSON string field value (full escape support, incl. \uXXXX and surrogate pairs)
std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = jFindKey(body, key);
    if (p == std::string::npos || p >= body.size() || body[p] != '"') return "";
    std::string r;
    for (p++; p < body.size() && body[p] != '"'; p++) {
        if (body[p] != '\\' || p + 1 >= body.size()) { r += body[p]; continue; }
        char e = body[++p];
        switch (e) {
            case 'n': r += '\n'; break;
            case 'r': r += '\r'; break;
            case 't': r += '\t'; break;
            case 'b': r += '\b'; break;
            case 'f': r += '\f'; break;
            case 'u': {
                int cp = hex4(body, p + 1);
                if (cp < 0) { r += 'u'; break; }
                p += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {                  // high surrogate: needs a low one
                    int lo = (p + 2 < body.size() && body[p+1] == '\\' && body[p+2] == 'u')
                             ? hex4(body, p + 3) : -1;
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p += 6;
                    } else cp = 0xFFFD;
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) cp = 0xFFFD; // lone low surrogate
                appendUtf8(r, (uint32_t)cp);
                break;
            }
            default: r += e; break;                                   // \" \\ \/
        }
    }
    return r;
}

// Extract a JSON integer field value
int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = jFindKey(body, key);
    if (p == std::string::npos) return def;
    try { return std::stoi(body.substr(p, 24)); } catch (...) { return def; }
}

bool parseBody(const std::string& b, std::string& meta,
               std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat  = extractStr(b, "category");
    emb.clear();
    size_t p = jFindKey(b, "embedding");
    if (p != std::string::npos && p < b.size() && b[p] == '[') {
        size_t e = b.find(']', p);
        if (e != std::string::npos) emb = parseVec(b.substr(p + 1, e - p - 1));
    }
    return !meta.empty() && !emb.empty();
}

// Cut to at most n bytes without splitting a UTF-8 character
std::string utf8Prefix(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
    return s.substr(0, n);
}

size_t countWords(const std::string& s) {
    size_t n = 0; bool in = false;
    for (unsigned char c : s) {
        bool sp = std::isspace(c);
        if (!sp && !in) n++;
        in = !sp;
    }
    return n;
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

std::vector<std::string> chunkText(const std::string& text,
                                   int chunkWords = 250, int overlapWords = 30)
{
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);

    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {text};

    std::vector<std::string> chunks;
    int step = chunkWords - overlapWords;
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) { if (j > i) chunk += ' '; chunk += words[j]; }
        chunks.push_back(chunk);
        if (end == (int)words.size()) break;
    }
    return chunks;
}

// =====================================================================
//  OLLAMA CLIENT  — wraps local Ollama REST API
//  Install:  https://ollama.com
//  Models:   ollama pull nomic-embed-text
//            ollama pull llama3.2
// =====================================================================

class OllamaClient {
    std::string host;
    int         port;

    // Parse {"embeddings":[[...],[...]]} from Ollama /api/embed response
    std::vector<std::vector<float>> parseEmbeddings(const std::string& body) {
        std::vector<std::vector<float>> out;
        size_t p = jFindKey(body, "embeddings");
        if (p == std::string::npos || body[p] != '[') return out;
        for (p++; p < body.size() && body[p] != ']'; p++) {
            if (body[p] != '[') continue;
            size_t e = body.find(']', p);
            if (e == std::string::npos) break;
            out.push_back(parseVec(body.substr(p + 1, e - p - 1)));
            p = e;
        }
        return out;
    }

    // Human-readable reason for a failed Ollama call
    std::string describe(const httplib::Result& res, const std::string& model) {
        if (!res) return "Ollama is not running. Start it with: ollama serve";
        auto e = extractStr(res->body, "error");
        if (e.find("not found") != std::string::npos)
            return "Ollama model '" + model + "' is missing. Run: ollama pull " + model;
        return "Ollama error (HTTP " + std::to_string(res->status) + ")" + (e.empty() ? "" : ": " + e);
    }

    // Parse {"response":"..."} from Ollama /api/generate response
    std::string parseResponse(const std::string& body) {
        return extractStr(body, "response");
    }

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3.2";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
        : host(h), port(p) {}

    bool isAvailable() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    // Embeds many texts, sending them to Ollama in batches.
    // Returns one vector per input, or an empty result with `err` set on failure.
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts,
                                               std::string& err, size_t batch = 32)
    {
        std::vector<std::vector<float>> out;
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(120, 0);
        for (size_t i = 0; i < texts.size(); i += batch) {
            std::string body = "{\"model\":" + jS(embedModel) + ",\"input\":[";
            for (size_t j = i; j < std::min(i + batch, texts.size()); j++)
                body += (j > i ? "," : "") + jS(texts[j]);
            body += "]}";
            auto res = cli.Post("/api/embed", body, "application/json");
            if (!res || res->status != 200) { err = describe(res, embedModel); return {}; }
            auto embs = parseEmbeddings(res->body);
            if (embs.size() != std::min(batch, texts.size() - i)) {
                err = "Ollama returned an unexpected embedding response"; return {};
            }
            for (auto& e : embs) out.push_back(std::move(e));
        }
        return out;
    }

    // Returns empty vector if Ollama is not running or model not found
    std::vector<float> embed(const std::string& text, std::string& err) {
        auto r = embedBatch({text}, err);
        return r.empty() ? std::vector<float>{} : r[0];
    }

    // Returns the model's answer, or an empty string with `err` set on failure
    std::string generate(const std::string& prompt, std::string& err) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);   // LLMs can be slow
        std::string body = "{\"model\":" + jS(genModel) + ","
                           "\"prompt\":" + jS(prompt) + ","
                           "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200) { err = describe(res, genModel); return ""; }
        return parseResponse(res->body);
    }
};

// =====================================================================
//  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
// =====================================================================

static const int MAX_DOC_CHUNKS = 2000;   // ~440k words per document
// Below this many chunks document search is an exact brute-force scan (a few ms).
// HNSW's simple "keep the closest M" neighbor pruning can leave outlier chunks
// (often the ones holding a specific fact) unreachable, which hurts RAG recall.
static const size_t EXACT_SEARCH_MAX = 20000;

struct DocItem {                 // one chunk of a document
    int         id;              // chunk id (node id in the indexes)
    int         docId;           // document it belongs to
    std::string title;           // e.g. "notes.pdf [2/9]"
    std::string text;
    std::vector<float> emb;
};

struct DocInfo {                 // one uploaded / pasted document
    int         docId;
    std::string title, kind, preview;
    int         chunks;
    size_t      words;
    std::vector<float> mapEmb;   // 16-D point for the scatter-plot map (may be empty)
};

// ── Binary document files: data/<user>/<docId>.doc ──────────────────
// "AURADOC1" | docId u32 | title | kind | words u64 | map (u32 n + n floats)
// | dims u32 | nChunks u32 | per chunk: text, dims floats
// Strings are u32 length + UTF-8 bytes. Numbers are little-endian (host order).
namespace docfile {
    static const char MAGIC[8] = {'A','U','R','A','D','O','C','1'};

    template <class T> void put(std::ostream& o, T v) { o.write((const char*)&v, sizeof v); }
    inline void putStr(std::ostream& o, const std::string& s) { put<uint32_t>(o, (uint32_t)s.size()); o.write(s.data(), s.size()); }
    inline void putFloats(std::ostream& o, const std::vector<float>& v) { o.write((const char*)v.data(), v.size() * sizeof(float)); }

    template <class T> bool get(std::istream& i, T& v) { return (bool)i.read((char*)&v, sizeof v); }
    inline bool getStr(std::istream& i, std::string& s, uint32_t max = 64u << 20) {
        uint32_t n; if (!get(i, n) || n > max) return false;
        s.resize(n); return (bool)i.read(&s[0], n);
    }
    inline bool getFloats(std::istream& i, std::vector<float>& v, uint32_t n) {
        v.resize(n); return (bool)i.read((char*)v.data(), n * sizeof(float));
    }

    bool write(const std::string& path, const DocInfo& info, const std::vector<std::string>& texts,
               const std::vector<std::vector<float>>& embs)
    {
        std::string tmp = path + ".tmp";
        {
            std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
            if (!o) return false;
            o.write(MAGIC, 8);
            put<uint32_t>(o, (uint32_t)info.docId);
            putStr(o, info.title); putStr(o, info.kind);
            put<uint64_t>(o, (uint64_t)info.words);
            put<uint32_t>(o, (uint32_t)info.mapEmb.size()); putFloats(o, info.mapEmb);
            put<uint32_t>(o, (uint32_t)(embs.empty() ? 0 : embs[0].size()));
            put<uint32_t>(o, (uint32_t)texts.size());
            for (size_t c = 0; c < texts.size(); c++) { putStr(o, texts[c]); putFloats(o, embs[c]); }
            if (!o.flush()) return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);                  // atomic replace
        return !ec;
    }

    bool read(const std::string& path, DocInfo& info, std::vector<std::string>& texts,
              std::vector<std::vector<float>>& embs)
    {
        std::ifstream i(path, std::ios::binary);
        char magic[8];
        uint32_t id, nMap, dims, n; uint64_t words;
        if (!i.read(magic, 8) || std::memcmp(magic, MAGIC, 8) != 0) return false;
        if (!get(i, id) || !getStr(i, info.title, 4096) || !getStr(i, info.kind, 64) || !get(i, words)) return false;
        if (!get(i, nMap) || nMap > 64 || !getFloats(i, info.mapEmb, nMap)) return false;
        if (!get(i, dims) || !get(i, n) || dims == 0 || dims > 8192 || n == 0 || n > (uint32_t)MAX_DOC_CHUNKS) return false;
        info.docId = (int)id; info.words = (size_t)words; info.chunks = (int)n;
        texts.resize(n); embs.resize(n);
        for (uint32_t c = 0; c < n; c++)
            if (!getStr(i, texts[c]) || !getFloats(i, embs[c], dims)) return false;
        return true;
    }
}

class DocumentDB {
    std::string dir;     // this user's save folder ("" = keep in memory only)
    std::unordered_map<int, DocItem> store;
    std::map<int, DocInfo> docs;
    HNSW       hnsw;
    BruteForce bf;       // exact search up to EXACT_SEARCH_MAX chunks
    std::mutex mu;
    int nextId    = 1;
    int nextDocId = 1;
    int dims      = 0;   // determined from first inserted embedding

    // HNSW::remove leaves holes in the graph, so after deleting a document
    // the indexes are rebuilt from the remaining chunks (in id order)
    void rebuildIndexes() {
        hnsw = HNSW(16, 200);
        bf   = BruteForce();
        std::vector<int> ids;
        for (auto& [id, it] : store) ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        for (int id : ids) {
            auto& it = store[id];
            VectorItem vi{id, it.title, "doc", it.emb};
            hnsw.insert(vi, cosine); bf.insert(vi);
        }
    }

    bool hasTitleLocked(const std::string& title) {
        for (auto& [id, d] : docs) if (d.title == title) return true;
        return false;
    }

public:
    bool hasTitle(const std::string& title) {
        std::lock_guard<std::mutex> lk(mu);
        return hasTitleLocked(title);
    }

private:
    // Adds a document's chunks to memory and the indexes (caller holds the lock)
    void addLocked(DocInfo info, const std::vector<std::string>& chunks,
                   const std::vector<std::vector<float>>& embs)
    {
        if (dims == 0) dims = (int)embs[0].size();
        for (size_t i = 0; i < chunks.size(); i++) {
            std::string chunkTitle = chunks.size() > 1
                ? info.title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : info.title;
            DocItem item{nextId++, info.docId, chunkTitle, chunks[i], embs[i]};
            VectorItem vi{item.id, chunkTitle, "doc", item.emb};
            hnsw.insert(vi, cosine); bf.insert(vi);
            store[item.id] = std::move(item);
        }
        std::string preview = utf8Prefix(chunks[0], 160);
        std::replace(preview.begin(), preview.end(), '\n', ' ');
        if (preview.size() < chunks[0].size()) preview += "…";
        info.preview = preview;
        info.chunks  = (int)chunks.size();
        docs[info.docId] = std::move(info);
        nextDocId = std::max(nextDocId, docs.rbegin()->first + 1);
    }

    std::string pathOf(int docId) { return dir + "/" + std::to_string(docId) + ".doc"; }

public:
    explicit DocumentDB(std::string saveDir = "") : dir(std::move(saveDir)), hnsw(16, 200) {}

    // Loads this user's saved documents from disk. Returns how many were loaded.
    int load() {
        namespace fs = std::filesystem;
        std::lock_guard<std::mutex> lk(mu);
        std::error_code ec;
        if (dir.empty() || !fs::is_directory(dir, ec)) return 0;
        std::vector<std::pair<int, std::string>> files;
        for (auto& e : fs::directory_iterator(dir, ec))
            if (e.path().extension() == ".doc") {
                try { files.push_back({std::stoi(e.path().stem().string()), e.path().string()}); } catch (...) {}
            }
        std::sort(files.begin(), files.end());
        int loaded = 0;
        for (auto& [id, path] : files) {
            DocInfo info; std::vector<std::string> texts; std::vector<std::vector<float>> embs;
            if (!docfile::read(path, info, texts, embs) || info.docId != id) {
                std::cerr << "WARNING: skipping unreadable document file " << path << std::endl; continue;
            }
            if (dims && (int)embs[0].size() != dims) {
                std::cerr << "WARNING: skipping " << path << " (embedding size " << embs[0].size()
                          << " != " << dims << "; was the embed model changed?)" << std::endl; continue;
            }
            addLocked(std::move(info), texts, embs);
            loaded++;
        }
        return loaded;
    }

    // Stores all chunks of a document at once (memory + disk). Returns docId, or -1 with `err` set.
    int insertDocument(const std::string& title, const std::string& kind,
                       const std::vector<std::string>& chunks,
                       const std::vector<std::vector<float>>& embs, size_t words,
                       const std::vector<float>& mapEmb, std::string& err)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (hasTitleLocked(title)) { err = "A document named \"" + title + "\" is already stored. Delete it first."; return -1; }
        int d = dims ? dims : (int)embs[0].size();
        for (auto& e : embs)
            if ((int)e.size() != d) {
                err = "Embedding size mismatch (" + std::to_string(e.size()) + " vs " + std::to_string(d) +
                      "). Did the embed model change? Restart the server.";
                return -1;
            }

        DocInfo info{nextDocId, title, kind, "", (int)chunks.size(), words, mapEmb};
        if (!dir.empty()) {                                      // save first: never keep an unsaved document
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
#ifndef _WIN32
            ::chmod(dir.c_str(), 0700);
#endif
            if (!docfile::write(pathOf(info.docId), info, chunks, embs)) {
                err = "Could not save the document to " + dir; return -1;
            }
        }
        addLocked(info, chunks, embs);
        return info.docId;
    }

    // Semantic search — returns top-k most similar chunks
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 0.7f)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty() || (int)q.size() != dims) return {};
        auto raw = (store.size() <= EXACT_SEARCH_MAX)
                   ? bf.knn(q, k, cosine)
                   : hnsw.knn(q, k, 50, cosine);
        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw)
            if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
        return out;
    }

    // Removes a whole document (all of its chunks)
    bool removeDocument(int docId) {
        std::lock_guard<std::mutex> lk(mu);
        if (!docs.count(docId)) return false;
        for (auto it = store.begin(); it != store.end(); )
            it = (it->second.docId == docId) ? store.erase(it) : std::next(it);
        docs.erase(docId);
        if (!dir.empty()) { std::error_code ec; std::filesystem::remove(pathOf(docId), ec); }
        rebuildIndexes();
        if (store.empty()) dims = 0;
        return true;
    }

    std::vector<DocInfo> list() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocInfo> r;
        for (auto& [id, d] : docs) r.push_back(d);
        return r;
    }

    size_t chunkCount() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    size_t docCount() {
        std::lock_guard<std::mutex> lk(mu);
        return docs.size();
    }

    int getDims() {
        std::lock_guard<std::mutex> lk(mu);
        return dims;
    }
};

// =====================================================================
//  DEMO DATA  (16D categorical vectors)
// =====================================================================

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");
    // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math",
        {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
}

// =====================================================================
//  AUTH — password hashing (PBKDF2-HMAC-SHA256), users, sessions
// =====================================================================

// SHA-256 (FIPS 180-4), self-contained so the build needs no crypto library
struct Sha256 {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t  buf[64]; size_t blen = 0; uint64_t total = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | p[4*i+1] << 16 | p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const uint8_t* p, size_t n) {
        total += n;
        while (n--) { buf[blen++] = *p++; if (blen == 64) { block(buf); blen = 0; } }
    }
    std::array<uint8_t, 32> digest() {
        uint64_t bits = total * 8;
        uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0; while (blen != 56) update(&z, 1);
        for (int i = 7; i >= 0; i--) { uint8_t b = (uint8_t)(bits >> (8 * i)); update(&b, 1); }
        std::array<uint8_t, 32> out;
        for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++) out[4*i+j] = (uint8_t)(h[i] >> (24 - 8*j));
        return out;
    }
};

using Bytes = std::vector<uint8_t>;

static std::array<uint8_t, 32> hmacSha256(const Bytes& key, const uint8_t* msg, size_t n) {
    uint8_t k[64] = {0};
    if (key.size() > 64) { Sha256 s; s.update(key.data(), key.size()); auto d = s.digest(); std::copy(d.begin(), d.end(), k); }
    else std::copy(key.begin(), key.end(), k);
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    Sha256 in; in.update(ipad, 64); in.update(msg, n); auto ih = in.digest();
    Sha256 out; out.update(opad, 64); out.update(ih.data(), 32);
    return out.digest();
}

// PBKDF2-HMAC-SHA256 with a 32-byte output (RFC 8018)
static Bytes pbkdf2(const std::string& password, const Bytes& salt, int iterations) {
    Bytes key(password.begin(), password.end()), msg(salt);
    msg.insert(msg.end(), {0, 0, 0, 1});                      // block index 1
    auto u = hmacSha256(key, msg.data(), msg.size());
    Bytes t(u.begin(), u.end());
    for (int i = 1; i < iterations; i++) {
        u = hmacSha256(key, u.data(), u.size());
        for (int j = 0; j < 32; j++) t[j] ^= u[j];
    }
    return t;
}

static Bytes randomBytes(size_t n) {
    Bytes b(n);
#ifdef _WIN32
    std::random_device rd;                                     // BCryptGenRandom on MinGW/MSVC
    for (auto& x : b) x = (uint8_t)rd();
#else
    std::ifstream f("/dev/urandom", std::ios::binary);
    f.read((char*)b.data(), (std::streamsize)n);
    if (!f) throw std::runtime_error("cannot read /dev/urandom");
#endif
    return b;
}

static std::string toHex(const Bytes& b) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (uint8_t x : b) { s += d[x >> 4]; s += d[x & 15]; }
    return s;
}

static Bytes fromHex(const std::string& s) {
    Bytes b;
    for (size_t i = 0; i + 1 < s.size(); i += 2) b.push_back((uint8_t)std::stoi(s.substr(i, 2), nullptr, 16));
    return b;
}

static bool constTimeEq(const Bytes& a, const Bytes& b) {
    if (a.size() != b.size()) return false;
    uint8_t r = 0;
    for (size_t i = 0; i < a.size(); i++) r |= a[i] ^ b[i];
    return r == 0;
}

// Registered users, persisted to users.txt as: <username> pbkdf2-sha256 <iterations> <salt hex> <hash hex>
class UserStore {
    struct User { int iterations; Bytes salt, hash; };
    struct Throttle { int fails = 0; std::chrono::steady_clock::time_point lockedUntil; };
    std::map<std::string, User> users;
    std::map<std::string, Throttle> throttle;
    std::string path;
    std::mutex mu;

    static const int ITERATIONS   = 200000;
    static const int MAX_FAILS    = 5;
    static const int LOCK_MINUTES = 15;

    void save() {                                   // write-then-rename so a crash can't truncate the file
        std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            for (auto& [name, u] : users)
                f << name << " pbkdf2-sha256 " << u.iterations << ' ' << toHex(u.salt) << ' ' << toHex(u.hash) << '\n';
        }
#ifndef _WIN32
        ::chmod(tmp.c_str(), 0600);
#endif
        std::remove(path.c_str());
        std::rename(tmp.c_str(), path.c_str());
    }

public:
    explicit UserStore(std::string p) : path(std::move(p)) {
        std::ifstream f(path);
        std::string name, algo, salt, hash; int it;
        while (f >> name >> algo >> it >> salt >> hash)
            if (algo == "pbkdf2-sha256") users[name] = {it, fromHex(salt), fromHex(hash)};
    }

    size_t count() { std::lock_guard<std::mutex> lk(mu); return users.size(); }

    static std::string validate(const std::string& name, const std::string& pw) {
        if (name.size() < 3 || name.size() > 32) return "Username must be 3–32 characters";
        for (char c : name)
            if (!std::isalnum((unsigned char)c) && c != '_' && c != '.' && c != '-')
                return "Username may only contain letters, digits, _ . -";
        if (pw.size() < 8)   return "Password must be at least 8 characters";
        if (pw.size() > 256) return "Password is too long";
        return "";
    }

    // Returns "" on success or an error message
    std::string create(const std::string& name, const std::string& pw) {
        auto err = validate(name, pw);
        if (!err.empty()) return err;
        auto salt = randomBytes(16);
        auto hash = pbkdf2(pw, salt, ITERATIONS);             // slow on purpose: done outside the lock
        std::lock_guard<std::mutex> lk(mu);
        if (users.count(name)) return "That username is taken";
        users[name] = {ITERATIONS, salt, hash};
        save();
        return "";
    }

    struct LoginResult {
        std::string err;         // "" on success; same message for unknown user / wrong password
        bool known   = false;    // username exists (for the server's log only, never sent to the client)
        bool blocked = false;    // rejected because the username is locked
        bool locked  = false;    // this failure triggered the lock
        int  fails   = 0;        // failures in the current window
    };
    static int maxFails() { return MAX_FAILS; }

    LoginResult check(const std::string& name, const std::string& pw) {
        LoginResult r;
        User u;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto& t = throttle[name];
            auto now = std::chrono::steady_clock::now();
            r.known = users.count(name) > 0;
            if (t.fails >= MAX_FAILS && now < t.lockedUntil) {
                r.blocked = true; r.fails = t.fails;
                r.err = "Too many failed attempts. Try again in " + std::to_string(LOCK_MINUTES) + " minutes.";
                return r;
            }
            u = r.known ? users[name] : User{ITERATIONS, Bytes(16, 0), Bytes(32, 0)};
        }
        // Hash even for unknown users so response time doesn't reveal which usernames exist
        bool ok = constTimeEq(pbkdf2(pw, u.salt, u.iterations), u.hash) && r.known;
        std::lock_guard<std::mutex> lk(mu);
        auto& t = throttle[name];
        if (ok) { t = Throttle(); return r; }
        if (t.fails >= MAX_FAILS) t.fails = 0;                 // previous lock expired: new window
        r.fails = ++t.fails;
        if (t.fails >= MAX_FAILS) {
            t.lockedUntil = std::chrono::steady_clock::now() + std::chrono::minutes(LOCK_MINUTES);
            r.locked = true;
        }
        r.err = "Invalid username or password";
        return r;
    }
};

// Signed-in sessions: random token (in an HttpOnly cookie) → username. In memory only.
class SessionStore {
    struct S { std::string user; std::chrono::steady_clock::time_point expires; };
    std::unordered_map<std::string, S> sessions;
    std::mutex mu;

public:
    static constexpr const char* COOKIE = "aura_session";
    static const int TTL_HOURS = 24 * 7;                       // sliding: renewed on every request

    std::string create(const std::string& user) {
        auto token = toHex(randomBytes(32));
        std::lock_guard<std::mutex> lk(mu);
        sessions[token] = {user, std::chrono::steady_clock::now() + std::chrono::hours(TTL_HOURS)};
        return token;
    }

    static std::string tokenOf(const httplib::Request& req) {
        auto c = req.get_header_value("Cookie");
        std::string key = std::string(COOKIE) + "=";
        for (size_t p = 0; p < c.size(); ) {
            while (p < c.size() && (c[p] == ' ' || c[p] == ';')) p++;
            size_t e = c.find(';', p);
            if (e == std::string::npos) e = c.size();
            if (c.compare(p, key.size(), key) == 0) return c.substr(p + key.size(), e - p - key.size());
            p = e;
        }
        return "";
    }

    // Username for this request, or "" if not signed in
    std::string user(const httplib::Request& req) {
        auto token = tokenOf(req);
        if (token.empty()) return "";
        std::lock_guard<std::mutex> lk(mu);
        auto it = sessions.find(token);
        if (it == sessions.end()) return "";
        auto now = std::chrono::steady_clock::now();
        if (now > it->second.expires) { sessions.erase(it); return ""; }
        it->second.expires = now + std::chrono::hours(TTL_HOURS);
        return it->second.user;
    }

    void destroy(const httplib::Request& req) {
        std::lock_guard<std::mutex> lk(mu);
        sessions.erase(tokenOf(req));
    }
};

// Sign-in activity log: printed to the terminal and appended to auth.log
//   2026-09-29 14:02:11  LOGIN OK       recruiter     127.0.0.1
class AuthLog {
    std::ofstream file;
    std::mutex mu;

    // Usernames in failed logins are whatever the client sent: keep one safe, short token per field
    static std::string clean(const std::string& s) {
        std::string o;
        for (unsigned char c : s) {
            if (o.size() >= 40) { o += "…"; break; }
            o += (c > 0x20 && c < 0x7F) ? (char)c : '?';
        }
        return o.empty() ? "-" : o;
    }

public:
    explicit AuthLog(const std::string& path) : file(path, std::ios::app) {
#ifndef _WIN32
        ::chmod(path.c_str(), 0600);
#endif
    }

    void write(const std::string& event, const std::string& user, const std::string& ip,
               const std::string& detail = "")
    {
        std::lock_guard<std::mutex> lk(mu);
        std::time_t t = std::time(nullptr);
        char when[32];
        std::strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        std::ostringstream line;
        line << when << "  " << std::left << std::setw(14) << event << ' '
             << std::setw(16) << clean(user) << ' ' << std::setw(15) << ip
             << (detail.empty() ? "" : "  " + detail);
        std::string out = line.str();
        out.erase(out.find_last_not_of(' ') + 1);
        std::cout << out << std::endl;
        file << out << '\n';
        file.flush();
    }
};

// Each user gets their own demo vectors and documents.
// Documents are saved under data/<username>/ and reloaded on first use after a restart.
struct UserSpace {
    VectorDB   db{DIMS};
    DocumentDB docs;
    std::mutex mapMu;
    std::unordered_map<int, int> mapPoint;   // docId → id of the document's dot on the demo map

    explicit UserSpace(const std::string& user) : docs("data/" + user) {
        loadDemo(db);
        int n = docs.load();
        for (auto& d : docs.list()) addMapPoint(d.docId, d.title, d.mapEmb);
        if (n) std::cout << "Loaded " << n << " saved document(s) for " << user << std::endl;
    }

    void addMapPoint(int docId, const std::string& title, const std::vector<float>& emb) {
        if ((int)emb.size() != DIMS) return;
        int id = db.insert(title, "doc", emb, getDistFn("cosine"));
        std::lock_guard<std::mutex> lk(mapMu);
        mapPoint[docId] = id;
    }

    void removeMapPoint(int docId) {
        int id = -1;
        {
            std::lock_guard<std::mutex> lk(mapMu);
            auto it = mapPoint.find(docId);
            if (it != mapPoint.end()) { id = it->second; mapPoint.erase(it); }
        }
        if (id >= 0) db.remove(id);
    }
};

// =====================================================================
//  HTTP SERVER
// =====================================================================

int main(int argc, char** argv) {
    // ./db [port] [--lan]     --lan also accepts connections from other devices on the network
    int  port = 8080;
    bool lan  = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--lan") lan = true;
        else if (std::isdigit((unsigned char)a[0])) port = std::atoi(a.c_str());
        else { std::cerr << "usage: ./db [port] [--lan]" << std::endl; return 1; }
    }

    OllamaClient ollama;
    UserStore    users("users.txt");
    SessionStore sessions;
    AuthLog      authLog("auth.log");

    std::map<std::string, std::unique_ptr<UserSpace>> spaces;
    std::mutex spacesMu;
    // Data space of the signed-in user (the auth gate below guarantees there is one)
    auto space = [&](const httplib::Request& req) -> UserSpace& {
        auto name = sessions.user(req);
        if (name.empty()) throw std::runtime_error("not signed in");
        std::lock_guard<std::mutex> lk(spacesMu);
        auto& sp = spaces[name];
        if (!sp) sp = std::make_unique<UserSpace>(name);
        return *sp;
    };

    // Check Ollama at startup (non-fatal)
    bool ollamaUp = ollama.isAvailable();
    std::cout << "=== Aura AI · VectorDB Engine ===" << std::endl;
    std::cout << "http://localhost:" << port
              << (lan ? "   (LAN access ON: other devices on your network can reach the login page)" : "   (this computer only)")
              << std::endl;
    std::cout << DIMS << "D demo vectors | HNSW+KD-Tree+BruteForce | "
              << users.count() << " registered user(s)" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)") << std::endl;
    if (ollamaUp) std::cout << "  embed model: " << ollama.embedModel
                            << "  gen model: "   << ollama.genModel << std::endl;

    httplib::Server svr;

    auto jsonErr = [](httplib::Response& res, int status, const std::string& msg) {
        res.status = status;
        res.set_content("{\"error\":" + jS(msg) + "}", "application/json");
    };

    // ── AUTH GATE ─────────────────────────────────────────────────────
    // Runs before every route and static file. Only the login page and
    // the auth endpoints are reachable without a valid session.
    static const std::set<std::string> PUBLIC = {
        "/login", "/auth/login", "/auth/register", "/auth/logout", "/favicon.ico" };
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (PUBLIC.count(req.path) || !sessions.user(req).empty())
            return httplib::Server::HandlerResponse::Unhandled;
        if (req.method == "GET" && req.path == "/") res.set_redirect("/login");
        else jsonErr(res, 401, "Not signed in");
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.set_exception_handler([&](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
        jsonErr(res, 500, "Internal error");
    });

    auto setSessionCookie = [](httplib::Response& res, const std::string& token, int maxAge) {
        res.set_header("Set-Cookie", std::string(SessionStore::COOKIE) + "=" + token +
                       "; Path=/; HttpOnly; SameSite=Strict; Max-Age=" + std::to_string(maxAge));
    };
    auto signIn = [&](httplib::Response& res, const std::string& name) {
        setSessionCookie(res, sessions.create(name), SessionStore::TTL_HOURS * 3600);
        res.set_content("{\"username\":" + jS(name) + "}", "application/json");
    };

    // POST /auth/register {"username":"...","password":"..."}  — creates the account and signs in
    svr.Post("/auth/register", [&](const httplib::Request& req, httplib::Response& res) {
        auto name = extractStr(req.body, "username"), pw = extractStr(req.body, "password");
        auto err = users.create(name, pw);
        if (!err.empty()) {
            authLog.write("SIGNUP FAILED", name, req.remote_addr, err);
            return jsonErr(res, err.find("taken") != std::string::npos ? 409 : 400, err);
        }
        authLog.write("SIGNUP", name, req.remote_addr, "new account, signed in");
        signIn(res, name);
    });

    // POST /auth/login {"username":"...","password":"..."}
    svr.Post("/auth/login", [&](const httplib::Request& req, httplib::Response& res) {
        auto name = extractStr(req.body, "username"), pw = extractStr(req.body, "password");
        auto r = users.check(name, pw);
        if (r.blocked) {
            authLog.write("LOGIN BLOCKED", name, req.remote_addr, "account is locked");
            return jsonErr(res, 429, r.err);
        }
        if (!r.err.empty()) {
            std::string detail = (r.known ? "wrong password" : "no such user") +
                std::string(" (") + std::to_string(r.fails) + "/" + std::to_string(UserStore::maxFails()) + ")";
            authLog.write("LOGIN FAILED", name, req.remote_addr, detail);
            if (r.locked) authLog.write("LOCKED", name, req.remote_addr, "too many failures, locked for 15 min");
            return jsonErr(res, 401, r.err);
        }
        authLog.write("LOGIN OK", name, req.remote_addr);
        signIn(res, name);
    });

    // POST /auth/logout
    svr.Post("/auth/logout", [&](const httplib::Request& req, httplib::Response& res) {
        auto name = sessions.user(req);
        if (!name.empty()) authLog.write("LOGOUT", name, req.remote_addr);
        sessions.destroy(req);
        setSessionCookie(res, "", 0);
        res.set_content("{\"ok\":true}", "application/json");
    });

    // GET /auth/me
    svr.Get("/auth/me", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content("{\"username\":" + jS(sessions.user(req)) + "}", "application/json");
    });

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        auto out = sp.db.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id = sp.db.insert(meta, cat, emb, getDistFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        int id  = std::stoi(req.matches[1]);
        bool ok = sp.db.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto items = sp.db.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":"        << v.id
               << ",\"metadata\":"  << jS(v.metadata)
               << ",\"category\":"  << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = sp.db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto gi = sp.db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert  {"title":"...","text":"...","kind":"PDF","map":[16 floats]}
    // "map" is the document's position on the demo scatter plot (optional).
    // The browser extracts text from uploaded files (PDF, Word, Excel, …) and sends it here.
    // Chunks the text, embeds all chunks via Ollama, then stores the document atomically.
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        auto kind  = extractStr(req.body, "kind");
        if (kind.empty()) kind = "TEXT";
        if (title.empty() || text.empty()) return jsonErr(res, 400, "need title and text");
        if (sp.docs.hasTitle(title))
            return jsonErr(res, 409, "A document named \"" + title + "\" is already stored. Delete it first.");

        auto chunks = chunkText(text, 250, 30);
        if (chunks.empty()) return jsonErr(res, 400, "The document contains no words");
        if ((int)chunks.size() > MAX_DOC_CHUNKS)
            return jsonErr(res, 413, "Document too large: " + std::to_string(chunks.size()) +
                           " chunks (max " + std::to_string(MAX_DOC_CHUNKS) + "). Split it into smaller files.");

        std::string err;
        auto embs = ollama.embedBatch(chunks, err);
        if (embs.empty()) return jsonErr(res, 503, err);

        std::vector<float> mapEmb;
        size_t mp = jFindKey(req.body, "map");
        if (mp != std::string::npos && req.body[mp] == '[') {
            size_t e = req.body.find(']', mp);
            if (e != std::string::npos) mapEmb = parseVec(req.body.substr(mp + 1, e - mp - 1));
        }

        size_t words = countWords(text);
        int docId = sp.docs.insertDocument(title, kind, chunks, embs, words, mapEmb, err);
        if (docId < 0) return jsonErr(res, err.rfind("Could not save", 0) == 0 ? 500 : 409, err);
        sp.addMapPoint(docId, title, mapEmb);

        std::ostringstream ss;
        ss << "{\"docId\":"  << docId
           << ",\"chunks\":" << chunks.size()
           << ",\"words\":"  << words
           << ",\"dims\":"   << sp.docs.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // DELETE /doc/delete/<docId>  — removes the whole document
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        int id  = std::stoi(req.matches[1]);
        bool ok = sp.docs.removeDocument(id);
        if (ok) sp.removeMapPoint(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    // GET /doc/list  — one entry per document
    svr.Get("/doc/list", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto docs = sp.docs.list();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            auto& d = docs[i];
            ss << "{\"docId\":"   << d.docId
               << ",\"title\":"   << jS(d.title)
               << ",\"kind\":"    << jS(d.kind)
               << ",\"chunks\":"  << d.chunks
               << ",\"words\":"   << d.words
               << ",\"preview\":" << jS(d.preview) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/search {"question":"...","k":3}
    // Fast retrieval for the UI visualizer
    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto question = extractStr(req.body, "question");
        int  k        = std::clamp(extractInt(req.body, "k", 3), 1, 10);
        if (question.empty()) return jsonErr(res, 400, "need question");

        std::string err;
        auto qEmb = ollama.embed(question, err);
        if (qEmb.empty()) return jsonErr(res, 503, err);

        auto hits = sp.docs.search(qEmb, k);

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"docId\":"    << hits[i].second.docId
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/ask  {"question":"...","k":3}
    // Full RAG pipeline: embed → retrieve → generate
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        auto question = extractStr(req.body, "question");
        int  k        = std::clamp(extractInt(req.body, "k", 3), 1, 10);
        if (question.empty()) return jsonErr(res, 400, "need question");

        // Step 1: embed the question
        std::string err;
        auto qEmb = ollama.embed(question, err);
        if (qEmb.empty()) return jsonErr(res, 503, err);

        // Step 2: retrieve top-k relevant chunks
        auto hits = sp.docs.search(qEmb, k);

        // Step 3: build prompt
        std::ostringstream ctx;
        for (int i = 0; i < (int)hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }
        std::string prompt =
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            "Just answer the question naturally.\n\n"
            "Context:\n" + ctx.str() +
            "Question: " + question + "\n\n"
            "Answer:";

        // Step 4: generate answer
        auto answer = ollama.generate(prompt, err);
        if (!err.empty()) return jsonErr(res, 503, err);

        // Step 5: return everything
        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(ollama.genModel)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"docId\":"    << hits[i].second.docId
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << sp.docs.docCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << jS(ollama.embedModel)
           << ",\"genModel\":"         << jS(ollama.genModel)
           << ",\"docCount\":"         << sp.docs.docCount()
           << ",\"chunkCount\":"       << sp.docs.chunkCount()
           << ",\"docDims\":"          << sp.docs.getDims()
           << ",\"demoDims\":"         << DIMS
           << ",\"demoCount\":"        << sp.db.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request& req, httplib::Response& res) {
        auto& sp = space(req);
        std::ostringstream ss;
        ss << "{\"count\":"      << sp.db.size()
           << ",\"dims\":"       << DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── STATIC FILES (served from the current directory) ──────────────
    auto serveFile = [](const std::string& path, const std::string& mime) {
        return [path, mime](const httplib::Request&, httplib::Response& res) {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) { res.status = 404; return; }
            res.set_content(std::string(std::istreambuf_iterator<char>(f),
                                        std::istreambuf_iterator<char>()), mime);
        };
    };
    svr.Get("/",           serveFile("index.html", "text/html; charset=utf-8"));
    svr.Get("/login", [&](const httplib::Request& req, httplib::Response& res) {
        if (!sessions.user(req).empty()) return res.set_redirect("/");
        serveFile("login.html", "text/html; charset=utf-8")(req, res);
    });
    svr.Get("/extract.js", serveFile("extract.js", "text/javascript; charset=utf-8"));
    // Third-party document parsers (pdf.js, mammoth, SheetJS, JSZip)
    if (!svr.set_mount_point("/vendor", "./vendor"))
        std::cout << "WARNING: ./vendor not found — run from the project folder or file uploads won't work" << std::endl;

    if (!svr.listen(lan ? "0.0.0.0" : "127.0.0.1", port)) {
        std::cerr << "ERROR: could not listen on port " << port
                  << " (already in use? try: ./db " << port + 1 << ")" << std::endl;
        return 1;
    }
    return 0;
}
