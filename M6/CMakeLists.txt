cmake_minimum_required(VERSION 3.24)
project(M6)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_win64_msvc16.zip
)

FetchContent_MakeAvailable(raylib)

add_executable(M6 M6.cpp)

target_link_libraries(M6 raylib winmm)

target_link_directories(M6 PRIVATE ${raylib_SOURCE_DIR}/lib)
target_include_directories(M6 PRIVATE ${raylib_SOURCE_DIR}/include)

# if(WIN32)
#     add_custom_command(TARGET M6 POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         $<TARGET_FILE:raylib>
#         $<TARGET_FILE_DIR:M6>
#     )
# endif()
