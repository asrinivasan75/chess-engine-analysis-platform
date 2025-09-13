cmake_minimum_required(VERSION 3.16)
project(chess_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_executable(chess_engine
  src/main.cpp
  src/uci.cpp
  src/board.cpp
  src/movegen.cpp
  src/search.cpp
  src/evaluate.cpp
  src/zobrist.cpp
)

target_include_directories(chess_engine PRIVATE include)
target_compile_options(chess_engine PRIVATE -O3 -DNDEBUG)
