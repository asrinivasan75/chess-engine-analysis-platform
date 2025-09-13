#pragma once
#include <cstdint>
#include <array>
#include "types.h"

namespace Zobrist {
    extern Key psq[PIECE_NB][SQUARE_NB];
    extern Key enpassant[FILE_NB];
    extern Key castling[CASTLING_RIGHT_NB];
    extern Key side;
    extern Key noPawns;
    
    void init();
    
    Key make_key(uint64_t seed);
}
