#pragma once
#include <vector>
#include <cassert>
#include "types.h"
#include "bitboards.h"

class Board;

enum GenType {
    CAPTURES,
    QUIETS,
    QUIET_CHECKS,
    EVASIONS,
    NON_EVASIONS,
    LEGAL
};

template<GenType>
ExtMove* generate(const Board& pos, ExtMove* moveList);

struct MoveList {
    explicit MoveList(const Board& pos);
    const ExtMove* begin() const { return moveList; }
    const ExtMove* end() const { return last; }
    size_t size() const { return last - moveList; }
    bool contains(Move move) const;

private:
    ExtMove moveList[MAX_MOVES], *last;
};

namespace Bitboards {
    void init();
    void pretty(Bitboard b);
    
    constexpr bool more_than_one(Bitboard b) {
        return b & (b - 1);
    }
    
    constexpr bool opposite_colors(Square s1, Square s2) {
        int s = int(s1) ^ int(s2);
        return ((s >> 3) ^ s) & 1;
    }
    
    extern uint8_t PopCnt16[1 << 16];
    extern uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];
    
    extern Bitboard SquareBB[SQUARE_NB];
    extern Bitboard FileBB[FILE_NB];
    extern Bitboard RankBB[RANK_NB];
    extern Bitboard AdjacentFilesBB[FILE_NB];
    extern Bitboard ForwardRanksBB[COLOR_NB][RANK_NB];
    extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
    extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
    extern Bitboard DistanceRingBB[SQUARE_NB][8];
    extern Bitboard ForwardFileBB[COLOR_NB][SQUARE_NB];
    extern Bitboard PassedPawnMask[COLOR_NB][SQUARE_NB];
    extern Bitboard PawnAttacksBB[COLOR_NB][SQUARE_NB];
    extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
    extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
}

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFF;
constexpr Bitboard Rank2BB = Rank1BB << (8 * 1);
constexpr Bitboard Rank3BB = Rank1BB << (8 * 2);
constexpr Bitboard Rank4BB = Rank1BB << (8 * 3);
constexpr Bitboard Rank5BB = Rank1BB << (8 * 4);
constexpr Bitboard Rank6BB = Rank1BB << (8 * 5);
constexpr Bitboard Rank7BB = Rank1BB << (8 * 6);
constexpr Bitboard Rank8BB = Rank1BB << (8 * 7);

constexpr Bitboard QueenSide   = FileABB | FileBBB | FileCBB | FileDBB;
constexpr Bitboard CenterFiles = FileCBB | FileDBB | FileEBB | FileFBB;
constexpr Bitboard KingSide    = FileEBB | FileFBB | FileGBB | FileHBB;
constexpr Bitboard Center      = (FileDBB | FileEBB) & (Rank4BB | Rank5BB);

constexpr Bitboard KingFlank[FILE_NB] = {
    QueenSide,   QueenSide, QueenSide,
    CenterFiles, CenterFiles,
    KingSide,    KingSide,  KingSide
};

extern int SquareDistance[SQUARE_NB][SQUARE_NB];

// Forward declarations
constexpr bool is_ok(Square s);
inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied);
inline Bitboard rook_attacks_bb(Square s, Bitboard occupied);

constexpr bool is_ok(Square s) {
    return s >= SQ_A1 && s <= SQ_H8;
}

inline Bitboard square_bb(Square s) {
    assert(is_ok(s));
    return Bitboards::SquareBB[s];
}

inline Bitboard make_bitboard(Square s) {
    return square_bb(s);
}

template<typename ...Squares>
inline Bitboard make_bitboard(Square s, Squares... squares) {
    return square_bb(s) | make_bitboard(squares...);
}

constexpr Bitboard shift(Bitboard b, int delta) {
    return  delta == 9 || delta == 1 || delta == -7 ? (b & ~FileHBB) << delta
          : delta == 7 || delta == -1 || delta == -9 ? (b & ~FileABB) << delta
          : b << delta;
}

template<int Delta>
constexpr Bitboard shift(Bitboard b) {
    if constexpr (Delta > 0) {
        return  Delta == 9 || Delta == 1 ? (b & ~FileHBB) << Delta
              : Delta == 7 ? (b & ~FileABB) << Delta
              : b << Delta;
    } else {
        constexpr int absDelta = -Delta;
        return  absDelta == 7 ? (b & ~FileHBB) >> absDelta
              : absDelta == 9 ? (b & ~FileABB) >> absDelta
              : absDelta == 1 ? (b & ~FileABB) >> absDelta
              : absDelta == 8 ? b >> absDelta
              : b >> absDelta;
    }
}

inline Bitboard pawn_attacks_bb(Color c, Square s) {
    assert(is_ok(s));
    return Bitboards::PawnAttacks[c][s];
}

inline Bitboard pawn_attacks_bb(Color c, Bitboard b) {
    return c == WHITE ? shift<9>(b & ~FileHBB) | shift<7>(b & ~FileABB)
                      : shift<-7>(b & ~FileHBB) | shift<-9>(b & ~FileABB);
}

inline Bitboard pawn_pushes_bb(Color c, Bitboard b, Bitboard empty) {
    return c == WHITE ? shift<8>(b) & empty : shift<-8>(b) & empty;
}

inline Bitboard pawn_double_pushes_bb(Color c, Bitboard b, Bitboard empty) {
    const Bitboard TRank = (c == WHITE ? Rank3BB : Rank6BB);
    return pawn_pushes_bb(c, pawn_pushes_bb(c, b, empty) & TRank, empty);
}

template<PieceType Pt>
inline Bitboard attacks_bb(Square s, Bitboard occupied) {
    assert(is_ok(s));
    
    switch (Pt)
    {
    case BISHOP: return bishop_attacks_bb(s, occupied);
    case ROOK  : return rook_attacks_bb(s, occupied);
    case QUEEN : return bishop_attacks_bb(s, occupied) | rook_attacks_bb(s, occupied);
    default    : return Bitboards::PseudoAttacks[Pt][s];
    }
}

template<>
inline Bitboard attacks_bb<PAWN>(Square s, Bitboard) {
    assert(false); // Explicit color required for pawn attacks
    return 0;
}

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {
    assert(is_ok(s));
    
    switch (pt)
    {
    case BISHOP: return bishop_attacks_bb(s, occupied);
    case ROOK  : return rook_attacks_bb(s, occupied);
    case QUEEN : return bishop_attacks_bb(s, occupied) | rook_attacks_bb(s, occupied);
    case KNIGHT: return Bitboards::PseudoAttacks[KNIGHT][s];
    case KING  : return Bitboards::PseudoAttacks[KING][s];
    default    : return 0;
    }
}

inline int popcount(Bitboard b) {
#ifndef USE_POPCNT
    union { Bitboard bb; uint16_t u[4]; } v = { b };
    return Bitboards::PopCnt16[v.u[0]] + Bitboards::PopCnt16[v.u[1]]
         + Bitboards::PopCnt16[v.u[2]] + Bitboards::PopCnt16[v.u[3]];
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
    return (int)_mm_popcnt_u64(b);
#else // Assumed gcc or compatible compiler
    return __builtin_popcountll(b);
#endif
}

inline Square lsb(Bitboard b) {
    assert(b);
#if defined(__GNUC__)  // GCC, Clang, ICC
    return Square(__builtin_ctzll(b));
#elif defined(_MSC_VER)  // MSVC
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return (Square) idx;
#else // Fallback
    const int index64[64] = {
         0, 47,  1, 56, 48, 27,  2, 60,
        57, 49, 41, 37, 28, 16,  3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11,  4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30,  9, 24,
        13, 18,  8, 12,  7,  6,  5, 63
    };
    const uint64_t debruijn64 = 0x03f566f7a88c5dd;
    return Square(index64[((b ^ (b-1)) * debruijn64) >> 58]);
#endif
}

inline Square msb(Bitboard b) {
    assert(b);
#if defined(__GNUC__)  // GCC, Clang, ICC
    return Square(63 ^ __builtin_clzll(b));
#elif defined(_MSC_VER)  // MSVC
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return (Square) idx;
#else // Fallback
    int result = 0;
    if (b > 0xFFFFFFFF) {
        b >>= 32;
        result = 32;
    }
    if (b > 0xFFFF) {
        b >>= 16;
        result += 16;
    }
    if (b > 0xFF) {
        b >>= 8;
        result += 8;
    }
    if (b > 0xF) {
        b >>= 4;
        result += 4;
    }
    if (b > 3) {
        b >>= 2;
        result += 2;
    }
    if (b > 1) {
        result += 1;
    }
    return Square(result);
#endif
}

inline Square pop_lsb(Bitboard& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline Square frontmost_sq(Color c, Bitboard b) {
    return c == WHITE ? msb(b) : lsb(b);
}

extern Bitboard bishop_attacks_bb(Square s, Bitboard occupied);
extern Bitboard rook_attacks_bb(Square s, Bitboard occupied);
