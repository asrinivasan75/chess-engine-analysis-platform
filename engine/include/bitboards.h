#pragma once
#include "types.h"

struct Bitboards {
    static Bitboard SquareBB[SQUARE_NB];
    static Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
    static Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
    static Bitboard LineBB[SQUARE_NB][SQUARE_NB];
    static Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
    
    static void init();
};

// Function declarations
inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied);
inline Bitboard rook_attacks_bb(Square s, Bitboard occupied);
inline Bitboard queen_attacks_bb(Square s, Bitboard occupied);

extern Bitboard RookMagics[SQUARE_NB];
extern Bitboard BishopMagics[SQUARE_NB];
extern Bitboard* RookAttacks[SQUARE_NB];
extern Bitboard* BishopAttacks[SQUARE_NB];

// File and rank bitboards
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