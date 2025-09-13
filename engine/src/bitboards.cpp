#include "bitboards.h"

// Global bitboard arrays
Bitboard Bitboards::SquareBB[SQUARE_NB];
Bitboard Bitboards::PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard Bitboards::PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard Bitboards::LineBB[SQUARE_NB][SQUARE_NB];
Bitboard Bitboards::BetweenBB[SQUARE_NB][SQUARE_NB];

Bitboard RookMagics[SQUARE_NB];
Bitboard BishopMagics[SQUARE_NB];
Bitboard* RookAttacks[SQUARE_NB];
Bitboard* BishopAttacks[SQUARE_NB];

void Bitboards::init() {
    // Initialize square bitboards
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        SquareBB[s] = 1ULL << s;
    }
    
    // Initialize pawn attacks
    for (Color c : {WHITE, BLACK}) {
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            int rank = s / 8;
            int file = s % 8;
            
            Bitboard attacks = 0;
            if (c == WHITE) {
                if (rank < 7) {
                    if (file > 0) attacks |= 1ULL << (s + 7);
                    if (file < 7) attacks |= 1ULL << (s + 9);
                }
            } else {
                if (rank > 0) {
                    if (file > 0) attacks |= 1ULL << (s - 9);
                    if (file < 7) attacks |= 1ULL << (s - 7);
                }
            }
            PawnAttacks[c][s] = attacks;
        }
    }
    
    // Initialize other piece attacks (simplified for now)
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        PseudoAttacks[KING][s] = 0; // TODO: Implement
        PseudoAttacks[KNIGHT][s] = 0; // TODO: Implement
        PseudoAttacks[BISHOP][s] = 0; // TODO: Implement
        PseudoAttacks[ROOK][s] = 0; // TODO: Implement
        PseudoAttacks[QUEEN][s] = 0; // TODO: Implement
    }
}

// Simple bishop attacks (without magic bitboards for now)
inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int rank = s / 8;
    int file = s % 8;
    
    // Diagonal directions
    int directions[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    
    for (int d = 0; d < 4; d++) {
        int dr = directions[d][0];
        int df = directions[d][1];
        
        for (int i = 1; i < 8; i++) {
            int newRank = rank + i * dr;
            int newFile = file + i * df;
            
            if (newRank < 0 || newRank > 7 || newFile < 0 || newFile > 7) break;
            
            Square target = newRank * 8 + newFile;
            attacks |= 1ULL << target;
            
            if (occupied & (1ULL << target)) break;
        }
    }
    return attacks;
}

// Simple rook attacks (without magic bitboards for now)
inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int rank = s / 8;
    int file = s % 8;
    
    // Horizontal and vertical directions
    int directions[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    
    for (int d = 0; d < 4; d++) {
        int dr = directions[d][0];
        int df = directions[d][1];
        
        for (int i = 1; i < 8; i++) {
            int newRank = rank + i * dr;
            int newFile = file + i * df;
            
            if (newRank < 0 || newRank > 7 || newFile < 0 || newFile > 7) break;
            
            Square target = newRank * 8 + newFile;
            attacks |= 1ULL << target;
            
            if (occupied & (1ULL << target)) break;
        }
    }
    return attacks;
}

inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return bishop_attacks_bb(s, occupied) | rook_attacks_bb(s, occupied);
}