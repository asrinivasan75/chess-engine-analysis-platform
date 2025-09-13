#include "board.h"
#include "zobrist.h"
#include "movegen.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>

Board::Board() {
    clear();
}

Board::Board(const std::string& fen) {
    set_fen(fen);
}

void Board::clear() {
    std::memset(this, 0, sizeof(Board));
    
    for (int i = 0; i < PIECE_TYPE_NB; ++i)
        byTypeBB[i] = 0;
    
    for (int i = 0; i < COLOR_NB; ++i)
        byColorBB[i] = 0;
    
    for (int i = 0; i < SQUARE_NB; ++i)
        board[i] = NO_PIECE;
    
    for (int i = 0; i < PIECE_NB; ++i)
        pieceCount[i] = 0;
    
    sideToMove = WHITE;
    gamePly = 0;
    chess960 = false;
    nodesSearched = 0;
    excludedMove = 0;
    
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::set_fen(const std::string& fenStr) {
    std::istringstream ss(fenStr);
    std::string board_str, side, castling, enpassant;
    int rule50, fullmove;
    
    ss >> board_str >> side >> castling >> enpassant >> rule50 >> fullmove;
    
    // Parse board
    int sq = SQ_A8;
    for (char c : board_str) {
        if (c == '/') {
            sq -= 16;
        } else if (std::isdigit(c)) {
            sq += c - '0';
        } else {
            Piece pc = NO_PIECE;
            switch (c) {
                case 'P': pc = W_PAWN; break;
                case 'N': pc = W_KNIGHT; break;
                case 'B': pc = W_BISHOP; break;
                case 'R': pc = W_ROOK; break;
                case 'Q': pc = W_QUEEN; break;
                case 'K': pc = W_KING; break;
                case 'p': pc = B_PAWN; break;
                case 'n': pc = B_KNIGHT; break;
                case 'b': pc = B_BISHOP; break;
                case 'r': pc = B_ROOK; break;
                case 'q': pc = B_QUEEN; break;
                case 'k': pc = B_KING; break;
            }
            if (pc != NO_PIECE) {
                put_piece(pc, Square(sq));
            }
            sq++;
        }
    }
    
    // Side to move
    sideToMove = (side == "w") ? WHITE : BLACK;
    
    // Initialize basic state
    gamePly = fullmove;
}

std::string Board::get_fen() const {
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}

Key Board::key() const {
    return st ? st->posKey : 0;
}

void Board::put_piece(Piece pc, Square s) {
    board[s] = pc;
    byTypeBB[type_of(pc)] |= square_bb(s);
    byColorBB[color_of(pc)] |= square_bb(s);
    pieceCount[pc]++;
    pieceList[pc][pieceCount[pc] - 1] = s;
    index[s] = pieceCount[pc] - 1;
}

void Board::remove_piece(Square s) {
    Piece pc = board[s];
    board[s] = NO_PIECE;
    byTypeBB[type_of(pc)] ^= square_bb(s);
    byColorBB[color_of(pc)] ^= square_bb(s);
    
    // Remove from piece list
    int lastIndex = pieceCount[pc] - 1;
    Square lastSquare = pieceList[pc][lastIndex];
    
    pieceList[pc][index[s]] = lastSquare;
    index[lastSquare] = index[s];
    
    pieceCount[pc]--;
}

void Board::move_piece(Square from, Square to) {
    Piece pc = board[from];
    remove_piece(from);
    put_piece(pc, to);
}

Bitboard Board::pieces() const {
    return byColorBB[WHITE] | byColorBB[BLACK];
}

Bitboard Board::pieces(Color c) const {
    return byColorBB[c];
}

Bitboard Board::pieces(PieceType pt) const {
    return byTypeBB[pt];
}

Bitboard Board::pieces(Color c, PieceType pt) const {
    return byColorBB[c] & byTypeBB[pt];
}

Bitboard Board::pieces(PieceType pt1, PieceType pt2) const {
    return byTypeBB[pt1] | byTypeBB[pt2];
}

Bitboard Board::pieces(Color c, PieceType pt1, PieceType pt2) const {
    return byColorBB[c] & (byTypeBB[pt1] | byTypeBB[pt2]);
}

Piece Board::piece_on(Square s) const {
    return board[s];
}

Square Board::ep_square() const {
    return st ? st->epSquare : SQ_NONE;
}

bool Board::empty(Square s) const {
    return board[s] == NO_PIECE;
}

bool Board::can_castle(CastlingRights cr) const {
    return st && (st->castlingRights & cr);
}

Color Board::side_to_move() const {
    return sideToMove;
}

int Board::game_ply() const {
    return gamePly;
}

bool Board::is_chess960() const {
    return chess960;
}

uint64_t Board::nodes_searched() const {
    return nodesSearched;
}

void Board::set_nodes_searched(uint64_t n) {
    nodesSearched = n;
}

bool Board::in_check() const {
    return false; // Simplified for now
}

Bitboard Board::checkers() const {
    return 0; // Simplified for now
}

Bitboard Board::attackers_to(Square s) const {
    return attackers_to(s, pieces());
}

Bitboard Board::attackers_to(Square s, Bitboard occupied) const {
    return 0; // Simplified for now
}

template<PieceType Pt> 
int Board::count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
}

template<PieceType Pt> 
Square Board::square(Color c) const {
    assert(count<Pt>(c) == 1);
    return pieceList[make_piece(c, Pt)][0];
}

// Explicit template instantiations
template int Board::count<PAWN>(Color) const;
template int Board::count<KNIGHT>(Color) const;
template int Board::count<BISHOP>(Color) const;
template int Board::count<ROOK>(Color) const;
template int Board::count<QUEEN>(Color) const;
template int Board::count<KING>(Color) const;

template Square Board::square<PAWN>(Color) const;
template Square Board::square<KNIGHT>(Color) const;
template Square Board::square<BISHOP>(Color) const;
template Square Board::square<ROOK>(Color) const;
template Square Board::square<QUEEN>(Color) const;
template Square Board::square<KING>(Color) const;

void Board::do_move(Move m, StateInfo& newSt) {
    // Simplified move implementation
    Key k = st ? st->posKey : 0;
    
    // Copy previous state
    newSt.previous = st;
    newSt.posKey = k;
    newSt.rule50 = st ? st->rule50 : 0;
    newSt.castlingRights = st ? st->castlingRights : 0;
    newSt.epSquare = SQ_NONE;
    newSt.capturedPiece = NO_PIECE;
    
    st = &newSt;
    gamePly++;
    
    if (m.from() != m.to()) {
        move_piece(m.from(), m.to());
    }
}

void Board::undo_move(Move m) {
    if (st && st->previous) {
        gamePly--;
        st = st->previous;
    }
}

bool Board::is_legal(Move m) const {
    return true; // Simplified
}

bool Board::gives_check(Move m) const {
    return false; // Simplified
}

bool Board::is_draw() const {
    return false; // Simplified
}

bool Board::is_mate() const {
    return false; // Simplified
}

bool Board::castling_impeded(CastlingRights cr) const {
    return false; // Simplified
}

Square Board::castling_rook_square(CastlingRights cr) const {
    return SQ_NONE; // Simplified
}

Bitboard Board::blockers_for_king(Color c) const {
    return 0; // Simplified
}

Bitboard Board::check_squares(PieceType pt) const {
    return 0; // Simplified
}

Key Board::key_after(Move m) const {
    return key(); // Simplified
}

Key Board::exclusion_key() const {
    return 0; // Simplified
}

Value Board::psq_score() const {
    return 0; // Simplified
}

Value Board::non_pawn_material(Color c) const {
    Value npm = 0;
    npm += count<KNIGHT>(c) * KnightValueMg;
    npm += count<BISHOP>(c) * BishopValueMg;
    npm += count<ROOK>(c) * RookValueMg;
    npm += count<QUEEN>(c) * QueenValueMg;
    return npm;
}

Value Board::non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
}

bool Board::opposite_bishops() const {
    return count<BISHOP>(WHITE) == 1 && count<BISHOP>(BLACK) == 1;
}

bool Board::is_repetition() const {
    return false; // Simplified
}
