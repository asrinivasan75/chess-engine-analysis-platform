#include "movegen.h"
#include "board.h"
#include <cstring>
#include <iostream>
#include <algorithm>

namespace Bitboards {
    uint8_t PopCnt16[1 << 16];
    uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];
    
    Bitboard SquareBB[SQUARE_NB];
    Bitboard FileBB[FILE_NB];
    Bitboard RankBB[RANK_NB];
    Bitboard AdjacentFilesBB[FILE_NB];
    Bitboard ForwardRanksBB[COLOR_NB][RANK_NB];
    Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
    Bitboard LineBB[SQUARE_NB][SQUARE_NB];
    Bitboard DistanceRingBB[SQUARE_NB][8];
    Bitboard ForwardFileBB[COLOR_NB][SQUARE_NB];
    Bitboard PassedPawnMask[COLOR_NB][SQUARE_NB];
    Bitboard PawnAttacksBB[COLOR_NB][SQUARE_NB];
    Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
    Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

    void init() {
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            SquareBB[s] = 1ULL << s;
            
        for (File f = FILE_A; f <= FILE_H; ++f)
            FileBB[f] = f > FILE_A ? FileBB[f - 1] << 1 : FileABB;
            
        for (Rank r = RANK_1; r <= RANK_8; ++r)
            RankBB[r] = Rank1BB << (8 * r);
            
        // Initialize popcount table
        for (int i = 0; i < (1 << 16); ++i)
            PopCnt16[i] = uint8_t(__builtin_popcount(i));
            
        // Initialize square distances
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
                SquareDistance[s1][s2] = std::max(
                    abs(file_of(s1) - file_of(s2)), 
                    abs(rank_of(s1) - rank_of(s2))
                );
                
        // Initialize pawn attacks
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            PawnAttacks[WHITE][s] = pawn_attacks_bb(WHITE, square_bb(s));
            PawnAttacks[BLACK][s] = pawn_attacks_bb(BLACK, square_bb(s));
        }
    }
    
    void pretty(Bitboard b) {
        std::cout << "+---+---+---+---+---+---+---+---+" << std::endl;
        for (Rank r = RANK_8; r >= RANK_1; --r) {
            for (File f = FILE_A; f <= FILE_H; ++f)
                std::cout << "| " << ((b & SquareBB[make_square(f, r)]) ? "X" : " ") << " ";
            std::cout << "|" << std::endl;
            std::cout << "+---+---+---+---+---+---+---+---+" << std::endl;
        }
        std::cout << std::endl;
    }
}

int SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return 0; // Simplified stub for now
}

Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return 0; // Simplified stub for now
}

MoveList::MoveList(const Board& pos) : last(moveList) {
    last = generate<LEGAL>(pos, last);
}

bool MoveList::contains(Move move) const {
    return std::find(begin(), end(), move) != end();
}

template<GenType Type>
ExtMove* generate(const Board& pos, ExtMove* moveList) {
    // Simplified move generation stub - just return a few basic moves
    if (Type == LEGAL || Type == NON_EVASIONS) {
        // Add a basic pawn move as example
        moveList->move = Move(SQ_E2, SQ_E4);
        moveList->score = 0;
        return moveList + 1;
    }
    return moveList;
}

// Explicit template instantiations
template ExtMove* generate<CAPTURES>(const Board&, ExtMove*);
template ExtMove* generate<QUIETS>(const Board&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Board&, ExtMove*);
template ExtMove* generate<QUIET_CHECKS>(const Board&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Board&, ExtMove*);
template ExtMove* generate<LEGAL>(const Board&, ExtMove*);
