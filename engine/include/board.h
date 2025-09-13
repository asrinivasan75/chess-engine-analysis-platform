#pragma once
#include <string>
#include <array>
#include <stack>
#include "types.h"

struct StateInfo {
    Key posKey;
    Value material;
    int nonPawnMaterial[COLOR_NB];
    int rule50;
    int pliesFromNull;
    int castlingRights;
    Square epSquare;
    Piece capturedPiece;
    StateInfo* previous;
};

class Board {
public:
    Board();
    Board(const std::string& fen);
    
    void set_fen(const std::string& fen);
    std::string get_fen() const;
    
    void do_move(Move m, StateInfo& newSt);
    void undo_move(Move m);
    
    bool is_legal(Move m) const;
    bool gives_check(Move m) const;
    bool in_check() const;
    bool is_draw() const;
    bool is_mate() const;
    
    Bitboard pieces() const;
    Bitboard pieces(Color c) const;
    Bitboard pieces(PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt) const;
    Bitboard pieces(PieceType pt1, PieceType pt2) const;
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const;
    
    Piece piece_on(Square s) const;
    Square ep_square() const;
    bool empty(Square s) const;
    
    template<PieceType Pt> int count(Color c) const;
    template<PieceType Pt> Square square(Color c) const;
    
    bool can_castle(CastlingRights cr) const;
    bool castling_impeded(CastlingRights cr) const;
    Square castling_rook_square(CastlingRights cr) const;
    
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    
    Key key() const;
    Key key_after(Move m) const;
    Key exclusion_key() const;
    
    Color side_to_move() const;
    int game_ply() const;
    bool is_chess960() const;
    
    uint64_t nodes_searched() const;
    void set_nodes_searched(uint64_t n);
    
    Value psq_score() const;
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;
    
    bool opposite_bishops() const;
    bool is_repetition() const;
    
    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    
private:
    Bitboard byTypeBB[PIECE_TYPE_NB];
    Bitboard byColorBB[COLOR_NB];
    Piece board[SQUARE_NB];
    int pieceCount[PIECE_NB];
    Square pieceList[PIECE_NB][16];
    int index[SQUARE_NB];
    
    int castlingRightsMask[SQUARE_NB];
    Square castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
    
    StateInfo* st;
    Color sideToMove;
    int gamePly;
    bool chess960;
    
    uint64_t nodesSearched;
    
    mutable Key excludedMove;
    
    void clear();
    void set_castling_right(Color c, Square rfrom);
    void set_state(StateInfo* si) const;
    void put_piece(Color c, PieceType pt, Square s);
    void remove_piece(Color c, PieceType pt, Square s);
    void move_piece(Color c, PieceType pt, Square from, Square to);
};
