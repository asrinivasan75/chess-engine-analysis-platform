#pragma once
#include "board.h"
#include "types.h"
#include <string>

namespace UCI {
    void init();
    void loop();
    
    std::string move_to_uci(Move m);
    Move uci_to_move(const std::string& str);
}
