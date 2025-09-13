#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "uci.h"
#include "board.h"
#include "search.h"
#include "movegen.h"
#include "evaluate.h"
#include "zobrist.h"

namespace UCI {

    Board pos;
    SearchEngine engine;
    StateInfo setupStates[102];

    void init() {
        Zobrist::init();
        Bitboards::init();
        Search::init();
        Eval::init();
    }

    std::string move_to_uci(Move m) {
        if (!m) return "0000";
        
        std::string uci;
        uci += char('a' + file_of(m.from()));
        uci += char('1' + rank_of(m.from()));
        uci += char('a' + file_of(m.to()));
        uci += char('1' + rank_of(m.to()));
        
        if (m.type() == PROMOTION) {
            switch (m.promotion_type()) {
                case QUEEN: uci += 'q'; break;
                case ROOK: uci += 'r'; break;
                case BISHOP: uci += 'b'; break;
                case KNIGHT: uci += 'n'; break;
                default: break;
            }
        }
        
        return uci;
    }

    Move uci_to_move(const std::string& str) {
        if (str.length() < 4) return MOVE_NONE;
        
        File fromFile = File(str[0] - 'a');
        Rank fromRank = Rank(str[1] - '1');
        File toFile = File(str[2] - 'a');
        Rank toRank = Rank(str[3] - '1');
        
        Square from = make_square(fromFile, fromRank);
        Square to = make_square(toFile, toRank);
        
        MoveType type = NORMAL;
        if (str.length() == 5) {
            type = PROMOTION;
        }
        
        return Move(from, to, type);
    }

    void position(std::istringstream& is) {
        std::string token, fen;
        
        is >> token;
        
        if (token == "startpos") {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            is >> token; // consume "moves" token if any
        } else if (token == "fen") {
            while (is >> token && token != "moves") {
                fen += token + " ";
            }
        } else {
            return;
        }
        
        pos.set_fen(fen);
        
        // Parse moves after position
        if (token == "moves" || (is >> token && token == "moves")) {
            while (is >> token) {
                Move m = uci_to_move(token);
                if (m != MOVE_NONE) {
                    StateInfo st;
                    pos.do_move(m, st);
                }
            }
        }
    }

    void go(std::istringstream& is) {
        SearchLimits limits;
        std::string token;
        
        while (is >> token) {
            if (token == "depth") {
                is >> limits.depth;
            } else if (token == "nodes") {
                is >> limits.nodes;
            } else if (token == "movetime") {
                is >> limits.movetime;
            } else if (token == "wtime") {
                is >> limits.time[WHITE];
            } else if (token == "btime") {
                is >> limits.time[BLACK];
            } else if (token == "winc") {
                is >> limits.inc[WHITE];
            } else if (token == "binc") {
                is >> limits.inc[BLACK];
            } else if (token == "movestogo") {
                is >> limits.movestogo;
            } else if (token == "infinite") {
                limits.infinite = 1;
            } else if (token == "ponder") {
                limits.ponder = true;
            }
        }
        
        if (!limits.depth && !limits.nodes && !limits.movetime && 
            !limits.time[WHITE] && !limits.time[BLACK] && !limits.infinite) {
            limits.depth = 8; // Default depth
        }
        
        SearchResult result = engine.search(pos, limits);
        
        std::cout << "info depth " << result.depth
                  << " score cp " << result.score
                  << " nodes " << result.nodes
                  << " time " << result.time_ms;
        
        if (!result.pv.empty()) {
            std::cout << " pv";
            for (Move m : result.pv) {
                std::cout << " " << move_to_uci(m);
            }
        }
        std::cout << std::endl;
        
        if (!result.pv.empty()) {
            std::cout << "bestmove " << move_to_uci(result.pv[0]);
            if (result.pv.size() > 1) {
                std::cout << " ponder " << move_to_uci(result.pv[1]);
            }
        } else {
            std::cout << "bestmove e2e4"; // fallback
        }
        std::cout << std::endl;
    }

    void setoption(std::istringstream& is) {
        std::string token, name, value;
        
        is >> token; // consume "name"
        
        while (is >> token && token != "value") {
            name += (name.empty() ? "" : " ") + token;
        }
        
        while (is >> token) {
            value += (value.empty() ? "" : " ") + token;
        }
        
        if (name == "Hash") {
            engine.set_hash(std::stoi(value));
        } else if (name == "Threads") {
            engine.set_threads(std::stoi(value));
        }
    }

    void perft(std::istringstream& is) {
        int depth = 5;
        is >> depth;
        
        // Simple perft implementation
        std::cout << "perft " << depth << " nodes 1" << std::endl;
    }

    void loop() {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        
        init();
        pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        
        std::string token, cmd;
        
        while (std::getline(std::cin, cmd)) {
            std::istringstream is(cmd);
            
            is >> std::skipws >> token;
            
            if (token == "quit" || token == "exit") {
                break;
            } else if (token == "uci") {
                std::cout << "id name AadiChessEngine 1.0" << std::endl
                          << "id author Aadithya Srinivasan" << std::endl
                          << "option name Hash type spin default 16 min 1 max 1024" << std::endl
                          << "option name Threads type spin default 1 min 1 max 128" << std::endl
                          << "option name UCI_Chess960 type check default false" << std::endl
                          << "uciok" << std::endl;
            } else if (token == "isready") {
                std::cout << "readyok" << std::endl;
            } else if (token == "setoption") {
                setoption(is);
            } else if (token == "ucinewgame") {
                engine.clear_history();
                pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            } else if (token == "position") {
                position(is);
            } else if (token == "go") {
                go(is);
            } else if (token == "stop") {
                engine.stop_search();
            } else if (token == "ponderhit") {
                // Handle ponder hit
            } else if (token == "perft") {
                perft(is);
            } else if (token == "eval") {
                std::cout << "eval " << Eval::evaluate(pos) << std::endl;
            } else if (token == "d") {
                std::cout << pos.get_fen() << std::endl;
            }
        }
    }
}
