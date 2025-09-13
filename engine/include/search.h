#pragma once
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <memory>
#include "board.h"
#include "types.h"

struct SearchLimits {
    SearchLimits() { clear(); }
    void clear() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = 
        movestogo = depth = nodes = mate = perft = infinite = 0;
        ponder = false;
    }
    
    std::vector<Move> searchmoves;
    int time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, movestogo,
        depth, nodes, mate, perft, infinite;
    bool ponder;
};

struct RootMove {
    explicit RootMove(Move m) : pv(1, m) {}
    bool extract_ponder_from_tt(Board& pos);
    bool operator==(const Move& m) const { return pv[0] == m; }
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score 
                               : m.previousScore < previousScore;
    }
    
    Value score = -VALUE_INFINITE;
    Value previousScore = -VALUE_INFINITE;
    int selDepth = 0;
    int tbRank = 0;
    Value tbScore;
    std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;

struct SearchInfo {
    std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;
    std::atomic<bool> stopOnPonderhit, stop, ponder;
    std::atomic<int> depth, selDepth, multiPV, iteration;
    Move bestMove, ponderMove;
    std::chrono::milliseconds optimumTime, maximumTime;
    int callsCnt;
    
    SearchInfo() { clear(); }
    void clear() {
        nodes = tbHits = bestMoveChanges = callsCnt = 0;
        depth = selDepth = multiPV = iteration = 0;
        optimumTime = maximumTime = std::chrono::milliseconds(0);
        stopOnPonderhit = stop = ponder = false;
        bestMove = ponderMove = MOVE_NONE;
    }
};

enum NodeType { NonPV, PV, Root };

struct TTEntry {
    Move move() const { return (Move)move16; }
    Value value() const { return (Value)value16; }
    Value eval() const { return (Value)eval16; }
    Depth depth() const { return (Depth)depth8 + DEPTH_OFFSET; }
    bool is_pv() const { return genBound8 & 0x4; }
    Bound bound() const { return (Bound)(genBound8 & 0x3); }
    void save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev);
    
private:
    friend class TranspositionTable;
    uint16_t key16;
    uint8_t depth8;
    uint8_t genBound8;
    int16_t value16;
    int16_t eval16;
    uint16_t move16;
};

class TranspositionTable {
    static constexpr int CacheLineSize = 64;
    static constexpr int ClusterSize = 3;
    
    struct Cluster {
        TTEntry entry[ClusterSize];
        char padding[2]; // Align to a divisor of the cache line size
    };
    
public:
    ~TranspositionTable() { delete[] table; }
    void new_search() { generation8 += 8; } // Lower 3 bits are used by PV flag and Bound
    TTEntry* probe(const Key key, bool& found) const;
    int hashfull() const;
    void resize(size_t mbSize);
    void clear();
    
    TTEntry* first_entry(const Key key) const {
        return &table[(size_t)key & (clusterCount - 1)].entry[0];
    }
    
private:
    friend struct TTEntry;
    
    size_t clusterCount;
    Cluster* table;
    uint8_t generation8; // Size must be not bigger than TTEntry::genBound8
};

extern TranspositionTable TT;

struct KillerMoves {
    Move killers[2];
    void update(Move m) {
        if (killers[0] != m) {
            killers[1] = killers[0];
            killers[0] = m;
        }
    }
    bool contains(Move m) const {
        return killers[0] == m || killers[1] == m;
    }
};

struct HistoryStats {
    static constexpr int Max = 1 << 28;
    
    const int& operator[](Piece pc, Square to) const {
        return table[pc][to];
    }
    int& operator[](Piece pc, Square to) {
        return table[pc][to];
    }
    void clear() { std::memset(table, 0, sizeof(table)); }
    
    void update(Piece pc, Square to, Value bonus) {
        int& entry = table[pc][to];
        int newEntry = entry + bonus * 32 - entry * abs(bonus) / 324;
        entry = std::clamp(newEntry, -Max, Max);
    }
    
private:
    int table[PIECE_NB][SQUARE_NB];
};

struct CounterMoveStats {
    const Move& operator[](Piece pc, Square to) const {
        return table[pc][to];
    }
    Move& operator[](Piece pc, Square to) {
        return table[pc][to];
    }
    void clear() { std::memset(table, 0, sizeof(table)); }
    
private:
    Move table[PIECE_NB][SQUARE_NB];
};

struct ContinuationHistory {
    const HistoryStats& operator[](bool inCheck, bool capture, Piece pc, Square to) const {
        return table[inCheck][capture][pc][to];
    }
    HistoryStats& operator[](bool inCheck, bool capture, Piece pc, Square to) {
        return table[inCheck][capture][pc][to];
    }
    void clear() { 
        for (auto& h : table)
            for (auto& i : h)
                for (auto& j : i)
                    for (auto& k : j)
                        k.clear(); 
    }
    
private:
    HistoryStats table[2][2][PIECE_NB][SQUARE_NB];
};

namespace Search {
    void init();
    void clear();
}

template<NodeType nodeType>
Value search(Board& pos, Value alpha, Value beta, Depth depth, bool cutNode);

template<NodeType nodeType>
Value qsearch(Board& pos, Value alpha, Value beta, Depth depth = 0);

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);

extern SearchLimits Limits;
extern LimitsType  limits;
extern SearchInfo SearchInfo;
extern RootMoves rootMoves;
extern StateInfo setupStates[102];
extern KillerMoves killers[MAX_PLY];
extern HistoryStats mainHistory;
extern CounterMoveStats counterMoves;
extern ContinuationHistory continuationHistory[2][2];

constexpr int DEPTH_OFFSET = -7; // To avoid negative depths
constexpr Depth DEPTH_QS_CHECKS     = 0;
constexpr Depth DEPTH_QS_NO_CHECKS  = -1;
constexpr Depth DEPTH_QS_RECAPTURES = -5;
constexpr Depth DEPTH_NONE = -6;

enum Bound { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = BOUND_UPPER | BOUND_LOWER };

Value futility_margin(Depth d, bool improving);

namespace Tablebases {
    void init(const std::string& paths);
    int probe_wdl(Board& pos, int* success);
    bool root_probe(Board& pos, RootMoves& rootMoves);
    bool root_probe_wdl(Board& pos, RootMoves& rootMoves);
}

struct SearchResult {
    Value score = VALUE_NONE;
    Depth depth = 0;
    std::vector<Move> pv;
    uint64_t nodes = 0;
    uint64_t time_ms = 0;
    std::string score_cp() const;
    std::string score_mate() const;
};

class SearchEngine {
public:
    SearchEngine();
    ~SearchEngine();
    
    SearchResult search(Board& pos, const SearchLimits& limits);
    void stop_search();
    void clear_history();
    void set_threads(int threads);
    void set_hash(int mb);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
