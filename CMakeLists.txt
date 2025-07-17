cmake_minimum_required(VERSION 3.15)
project(AISnake)

set(CMAKE_CXX_STANDARD 17)

if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    
    # Combine all linker flags into a single set command to avoid duplication
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} \
        -sUSE_SDL=2 \
        -sUSE_WEBGL2=1 \
        -sALLOW_MEMORY_GROWTH=1 \
        -sASSERTIONS=1 \
        -sINITIAL_MEMORY=64MB \
        -sMAX_WEBGL_VERSION=2 \
        -sEXPORT_ALL=1 \
        --shell-file ${CMAKE_SOURCE_DIR}/shell.html \
        -sEXPORTED_FUNCTIONS=['_main','_getExplorationRate','_malloc','_free'] \
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','allocate','intArrayFromString'] \
        -sMODULARIZE=1 \
        -sENVIRONMENT=web,worker \
        -sNO_EXIT_RUNTIME=1 \
        -sINVOKE_RUN=0 \
        -sSTRICT=1")
        
    # Compiler flags
    add_compile_options(-O3 -flto -fno-rtti -fno-exceptions)
    
    # For debugging (uncomment when needed)
    # add_compile_options(-g -sSAFE_HEAP=1 -sSTACK_OVERFLOW_CHECK=1)
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sDEMANGLE_SUPPORT=1")
else()
    find_package(SDL2 REQUIRED)
    add_compile_options(-O3)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

add_executable(aisnake_web src/main.cpp)

if(NOT EMSCRIPTEN)
    target_link_libraries(aisnake_web ${SDL2_LIBRARIES})
endif()

# For Emscripten, set output names
if(EMSCRIPTEN)
    set_target_properties(aisnake_web PROPERTIES
        OUTPUT_NAME "aisnake_web"
        SUFFIX ".html"
    )
endif()
