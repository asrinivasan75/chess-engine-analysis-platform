#include "zobrist.h"
#include <random>

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side;
Key noPawns;

Key make_key(uint64_t seed) {
    return seed * 6364136223846793005ULL + 1442695040888963407ULL;
}

void init() {
    std::mt19937_64 rng(1070372);
    
    for (Piece pc : {NO_PIECE, W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                     B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING})
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            psq[pc][s] = rng();
            
    for (File f = FILE_A; f <= FILE_H; ++f)
        enpassant[f] = rng();
        
    for (int cf = NO_CASTLING; cf <= ANY_CASTLING; ++cf) {
        Key k = 0;
        Bitboard b = cf;
        while (b) {
            Key r = rng();
            if (cf & WHITE_OO)
                k ^= (cf & WHITE_OO) ? r : 0;
            if (cf & WHITE_OOO) 
                k ^= (cf & WHITE_OOO) ? r : 0;
            if (cf & BLACK_OO)
                k ^= (cf & BLACK_OO) ? r : 0;  
            if (cf & BLACK_OOO)
                k ^= (cf & BLACK_OOO) ? r : 0;
            b &= b - 1;
        }
        castling[cf] = k;
    }
    
    side = rng();
    noPawns = rng();
}

}
