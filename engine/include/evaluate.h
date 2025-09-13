#pragma once
#include "board.h"
#include "types.h"
#include <string>

namespace Eval {
    std::string trace(const Board& pos);
    Value evaluate(const Board& pos);
    void init();
    
    extern bool useNNUE;
    extern std::string currentEvalFileName;
}

namespace NNUE {
    void init();
    bool load_eval(const std::string& name, std::istream& stream);
    bool save_eval(std::ostream& stream);
    Value evaluate(const Board& pos);
    Value compute_eval(const Board& pos);
    void verify();
}

struct Score {
    int mg, eg;
    
    constexpr Score(int mg = 0, int eg = 0) : mg(mg), eg(eg) {}
    Score operator+(Score s) const { return {mg + s.mg, eg + s.eg}; }
    Score operator-(Score s) const { return {mg - s.mg, eg - s.eg}; }
    Score operator*(int i) const { return {mg * i, eg * i}; }
    Score& operator+=(Score s) { mg += s.mg; eg += s.eg; return *this; }
    Score& operator-=(Score s) { mg -= s.mg; eg -= s.eg; return *this; }
    Score& operator*=(int i) { mg *= i; eg *= i; return *this; }
    Score operator-() const { return {-mg, -eg}; }
};

constexpr Score make_score(int mg, int eg) {
    return Score(mg, eg);
}

constexpr Value mg_value(Score s) {
    return Value(s.mg);
}

constexpr Value eg_value(Score s) {
    return Value(s.eg);
}

inline Value evaluate_scale_factor(const Board& pos, Value eval) {
    return eval;
}

namespace PieceSquare {
    extern Score psq[PIECE_NB][SQUARE_NB];
    void init();
}

namespace Pawns {
    struct Entry {
        Score pawnScore(Color c) const { return scores[c]; }
        Bitboard pawn_attacks(Color c) const { return pawnAttacks[c]; }
        Bitboard passed_pawns(Color c) const { return passedPawns[c]; }
        Bitboard pawn_attacks_span(Color c) const { return pawnAttacksSpan[c]; }
        int weak_unopposed(Color c) const { return weakUnopposed[c]; }
        int passed_count() const { return passedCount; }
        
        template<Color Us> Score king_safety(const Board& pos, Square ksq);
        template<Color Us> Value shelter_storm(const Board& pos, Square ksq);
        
        Key key;
        Score scores[COLOR_NB];
        Bitboard passedPawns[COLOR_NB];
        Bitboard pawnAttacks[COLOR_NB];
        Bitboard pawnAttacksSpan[COLOR_NB];
        int weakUnopposed[COLOR_NB];
        int castlingRights[COLOR_NB];
        int passedCount;
    };
    
    struct Table {
        Entry* probe(const Board& pos) { return &entries[0]; } // Simplified
        Entry entries[16384];
    };
    
    Entry* probe(const Board& pos);
    template<Color Us> Score evaluate(const Board& pos, Entry* e);
}

namespace Material {
    struct Entry {
        Score material_imbalance() const { return make_score(value, value); }
        Value space_weight() const { return spaceWeight; }
        Phase game_phase() const { return gamePhase; }
        bool specialized_eval_exists() const { return evaluationFunction != nullptr; }
        Value evaluate(const Board& pos) const;
        
        int16_t value;
        uint8_t factor[COLOR_NB];
        Key key;
        int16_t spaceWeight;
        Phase gamePhase;
        uint8_t scalingFunction[COLOR_NB];
        Value (*evaluationFunction)(const Board&);
    };
    
    struct Table {
        Entry* probe(const Board& pos) { return &entries[0]; } // Simplified
        Entry entries[8192];
    };
    
    Entry* probe(const Board& pos);
    Phase game_phase(const Board& pos);
    void init();
}

template<Color Us>
Value evaluate(const Board& pos) {
    return Us == WHITE ? Eval::evaluate(pos) : -Eval::evaluate(pos);
}

constexpr int PawnValueMg   = 126,   PawnValueEg   = 208;
constexpr int KnightValueMg = 781,   KnightValueEg = 854;
constexpr int BishopValueMg = 825,   BishopValueEg = 915;
constexpr int RookValueMg   = 1276,  RookValueEg   = 1380;
constexpr int QueenValueMg  = 2538,  QueenValueEg  = 2682;

constexpr Score PieceValue[PHASE_NB][PIECE_TYPE_NB] = {
    { VALUE_ZERO, make_score(PawnValueMg, PawnValueEg),
      make_score(KnightValueMg, KnightValueEg),
      make_score(BishopValueMg, BishopValueEg),
      make_score(RookValueMg, RookValueEg),
      make_score(QueenValueMg, QueenValueEg) },
    { VALUE_ZERO, make_score(PawnValueMg, PawnValueEg),
      make_score(KnightValueMg, KnightValueEg),
      make_score(BishopValueMg, BishopValueEg),
      make_score(RookValueMg, RookValueEg),
      make_score(QueenValueMg, QueenValueEg) }
};

enum Phase { PHASE_ENDGAME, PHASE_MIDGAME, PHASE_NB };

constexpr int MidgameLimit  = 15258;
constexpr int EndgameLimit  = 3915;

inline Phase game_phase(const Board& pos) {
    Value npm = pos.non_pawn_material(WHITE) + pos.non_pawn_material(BLACK);
    return npm >= MidgameLimit ? PHASE_MIDGAME
         : npm <= EndgameLimit ? PHASE_ENDGAME
         : Phase(((npm - EndgameLimit) * PHASE_MIDGAME) / (MidgameLimit - EndgameLimit));
}

inline Value interpolate(const Score& s, Phase ph) {
    return Value(mg_value(s) * ph + eg_value(s) * (PHASE_MIDGAME - ph)) / PHASE_MIDGAME;
}
