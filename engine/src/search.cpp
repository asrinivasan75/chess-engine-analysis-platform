#include "search.h"
#include "evaluate.h"
#include "movegen.h"
#include <memory>
#include <chrono>
#include <atomic>

// Global search data
TranspositionTable TT;
SearchLimits Limits;
RootMoves rootMoves;
StateInfo setupStates[102];
KillerMoves killers[MAX_PLY];
HistoryStats mainHistory;
CounterMoveStats counterMoves;
ContinuationHistory continuationHistory[2][2];

namespace Search {
    void init() {
        TT.resize(16); // 16MB default
        mainHistory.clear();
        counterMoves.clear();
        for (auto& ch : continuationHistory)
            for (auto& ch2 : ch)
                ch2.clear();
    }
    
    void clear() {
        TT.clear();
        mainHistory.clear();
        counterMoves.clear();
        for (int i = 0; i < MAX_PLY; ++i) {
            killers[i].killers[0] = killers[i].killers[1] = MOVE_NONE;
        }
    }
}

// Transposition Table implementation
void TranspositionTable::resize(size_t mbSize) {
    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);
    if (table) delete[] table;
    table = new Cluster[clusterCount]();
    clear();
}

void TranspositionTable::clear() {
    std::memset(table, 0, clusterCount * sizeof(Cluster));
}

TTEntry* TranspositionTable::probe(const Key key, bool& found) const {
    TTEntry* tte = first_entry(key);
    const uint16_t key16 = key >> 48;
    
    for (int i = 0; i < ClusterSize; ++i)
        if (!tte[i].key16 || tte[i].key16 == key16) {
            tte[i].key16 = key16;
            found = (bool)tte[i].depth8;
            return &tte[i];
        }
    
    // Replace strategy
    TTEntry* replace = tte;
    for (int i = 1; i < ClusterSize; ++i)
        if (replace->depth8 - replace->genBound8 > tte[i].depth8 - tte[i].genBound8)
            replace = &tte[i];
    
    found = false;
    return replace;
}

void TTEntry::save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev) {
    // Simplified implementation
    key16 = (uint16_t)(k >> 48);
    value16 = (int16_t)v;
    eval16 = (int16_t)ev;
    depth8 = (uint8_t)(d + DEPTH_OFFSET);
    genBound8 = (uint8_t)(TT.generation8 | (pv ? 4 : 0) | b);
    move16 = (uint16_t)m.data;
}

int TranspositionTable::hashfull() const {
    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < ClusterSize; ++j)
            cnt += table[i].entry[j].depth8 && 
                  (table[i].entry[j].genBound8 & 0xF8) == generation8;
    return cnt / ClusterSize;
}

// Search engine implementation
class SearchEngine::Impl {
public:
    std::atomic<bool> stopSearch{false};
    int threads = 1;
    
    SearchResult search(Board& pos, const SearchLimits& limits) {
        auto start = std::chrono::steady_clock::now();
        
        SearchResult result;
        result.depth = limits.depth > 0 ? limits.depth : 8;
        result.score = Eval::evaluate(pos);
        result.nodes = 1000; // Dummy value
        result.pv.push_back(Move(SQ_E2, SQ_E4)); // Dummy move
        
        auto end = std::chrono::steady_clock::now();
        result.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        return result;
    }
    
    void stop() { stopSearch = true; }
    void clear_history() { Search::clear(); }
    void set_threads(int t) { threads = t; }
    void set_hash(int mb) { TT.resize(mb); }
};

SearchEngine::SearchEngine() : pImpl(std::make_unique<Impl>()) {}
SearchEngine::~SearchEngine() = default;

SearchResult SearchEngine::search(Board& pos, const SearchLimits& limits) {
    return pImpl->search(pos, limits);
}

void SearchEngine::stop_search() { pImpl->stop(); }
void SearchEngine::clear_history() { pImpl->clear_history(); }
void SearchEngine::set_threads(int threads) { pImpl->set_threads(threads); }
void SearchEngine::set_hash(int mb) { pImpl->set_hash(mb); }

std::string SearchResult::score_cp() const {
    return std::to_string(score);
}

std::string SearchResult::score_mate() const {
    if (abs(score) >= VALUE_MATE - 100)
        return std::to_string((VALUE_MATE - abs(score) + 1) / 2 * (score > 0 ? 1 : -1));
    return score_cp();
}

// Simple alpha-beta search stub
template<NodeType nodeType>
Value search(Board& pos, Value alpha, Value beta, Depth depth, bool cutNode) {
    if (depth <= 0)
        return qsearch<nodeType>(pos, alpha, beta, 0);
        
    return Eval::evaluate(pos);
}

template<NodeType nodeType>
Value qsearch(Board& pos, Value alpha, Value beta, Depth depth) {
    return Eval::evaluate(pos);
}

Value value_to_tt(Value v, int ply) {
    return v >= VALUE_MATE - 100 ?  v + ply
         : v <= -VALUE_MATE + 100 ? v - ply : v;
}

Value value_from_tt(Value v, int ply, int r50c) {
    return v == VALUE_NONE ? VALUE_NONE
         : v >= VALUE_MATE - 100 ?  v - ply
         : v <= -VALUE_MATE + 100 ? v + ply : v;
}

Value futility_margin(Depth d, bool improving) {
    return Value(154 - 32 * improving + 144 * d);
}

// Template instantiations
template Value search<PV>(Board&, Value, Value, Depth, bool);
template Value search<NonPV>(Board&, Value, Value, Depth, bool);
template Value qsearch<PV>(Board&, Value, Value, Depth);
template Value qsearch<NonPV>(Board&, Value, Value, Depth);
