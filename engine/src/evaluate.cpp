#include "evaluate.h"
#include "board.h"
#include <algorithm>
#include <sstream>

namespace Eval {
    bool useNNUE = false;
    std::string currentEvalFileName = "nn-default.nnue";
    
    void init() {
        PieceSquare::init();
        // NNUE::init(); // Would initialize NNUE if available
    }
    
    Value evaluate(const Board& pos) {
        if (useNNUE) {
            return NNUE::evaluate(pos);
        }
        
        // Classical evaluation with tapered evaluation
        Score score = {0, 0};
        
        // Material evaluation
        for (Color c = WHITE; c <= BLACK; ++c) {
            for (PieceType pt = PAWN; pt <= KING; ++pt) {
                int count = pos.count<PAWN>(c); // Simplified - should iterate all piece types
                if (pt == PAWN) score += PieceValue[PHASE_MIDGAME][pt] * pos.count<PAWN>(c) * (c == WHITE ? 1 : -1);
                else if (pt == KNIGHT) score += PieceValue[PHASE_MIDGAME][pt] * pos.count<KNIGHT>(c) * (c == WHITE ? 1 : -1);
                else if (pt == BISHOP) score += PieceValue[PHASE_MIDGAME][pt] * pos.count<BISHOP>(c) * (c == WHITE ? 1 : -1);
                else if (pt == ROOK) score += PieceValue[PHASE_MIDGAME][pt] * pos.count<ROOK>(c) * (c == WHITE ? 1 : -1);
                else if (pt == QUEEN) score += PieceValue[PHASE_MIDGAME][pt] * pos.count<QUEEN>(c) * (c == WHITE ? 1 : -1);
            }
        }
        
        // Piece-square table evaluation
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            Piece pc = pos.piece_on(s);
            if (pc != NO_PIECE) {
                score += PieceSquare::psq[pc][s] * (color_of(pc) == WHITE ? 1 : -1);
            }
        }
        
        // Tapered evaluation between midgame and endgame
        Phase ph = game_phase(pos);
        Value eval = interpolate(score, ph);
        
        // Return from side-to-move perspective
        return pos.side_to_move() == WHITE ? eval : -eval;
    }
    
    std::string trace(const Board& pos) {
        std::ostringstream ss;
        ss << "Material: " << pos.non_pawn_material(WHITE) - pos.non_pawn_material(BLACK) << std::endl;
        ss << "PSQ: " << pos.psq_score() << std::endl;
        ss << "Total: " << evaluate(pos) << std::endl;
        return ss.str();
    }
}

namespace NNUE {
    void init() {
        // Initialize NNUE network - stub implementation
    }
    
    bool load_eval(const std::string& name, std::istream& stream) {
        // Load NNUE evaluation from stream
        return false; // Not implemented
    }
    
    bool save_eval(std::ostream& stream) {
        // Save NNUE evaluation to stream
        return false; // Not implemented
    }
    
    Value evaluate(const Board& pos) {
        // NNUE evaluation - simplified stub
        return 25; // Small positive evaluation
    }
    
    Value compute_eval(const Board& pos) {
        return evaluate(pos);
    }
    
    void verify() {
        // Verify NNUE network integrity
    }
}

namespace PieceSquare {
    Score psq[PIECE_NB][SQUARE_NB];
    
    void init() {
        // Initialize piece-square tables with basic values
        
        // Pawn piece-square table values (simplified)
        const Score PawnPSQ[RANK_NB] = {
            make_score(  0,   0), make_score( 10,  10), make_score( 20,  20), make_score( 30,  30),
            make_score( 40,  40), make_score( 50,  50), make_score( 60,  60), make_score(  0,   0)
        };
        
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            psq[NO_PIECE][s] = make_score(0, 0);
            
            // Pawns
            psq[W_PAWN][s] = PawnPSQ[rank_of(s)];
            psq[B_PAWN][s] = PawnPSQ[RANK_8 - rank_of(s)];
            
            // Knights - centralization bonus
            int knight_bonus = 4 - std::max(abs(file_of(s) - FILE_D), abs(rank_of(s) - RANK_4));
            psq[W_KNIGHT][s] = make_score(knight_bonus * 5, knight_bonus * 3);
            psq[B_KNIGHT][s] = make_score(knight_bonus * 5, knight_bonus * 3);
            
            // Other pieces - simplified
            psq[W_BISHOP][s] = make_score(0, 0);
            psq[B_BISHOP][s] = make_score(0, 0);
            psq[W_ROOK][s] = make_score(0, 0);
            psq[B_ROOK][s] = make_score(0, 0);
            psq[W_QUEEN][s] = make_score(0, 0);
            psq[B_QUEEN][s] = make_score(0, 0);
            
            // Kings
            psq[W_KING][s] = make_score(-30, 0);
            psq[B_KING][s] = make_score(-30, 0);
        }
    }
}

namespace Material {
    void init() {
        // Initialize material evaluation tables
    }
    
    Phase game_phase(const Board& pos) {
        return ::game_phase(pos);
    }
}
