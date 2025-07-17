cmake_minimum_required(VERSION 3.15)
project(AISnake)

set(CMAKE_CXX_STANDARD 17)

if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    
    # SDL2 setup for Emscripten
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
    
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} \
        -sUSE_SDL=2 \
        -sUSE_WEBGL2=1 \
        -sALLOW_MEMORY_GROWTH=1 \
        -sASSERTIONS=1 \
        -sINITIAL_MEMORY=128MB \
        --shell-file ${CMAKE_SOURCE_DIR}/shell.html \
        -sEXPORTED_FUNCTIONS=['_main','_getExplorationRate','_malloc','_free'] \
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString'] \
        -sMODULARIZE=1 \
        -sENVIRONMENT=web \
        -sNO_EXIT_RUNTIME=1 \
        -sFORCE_FILESYSTEM=1 \
        -sDYNAMIC_EXECUTION=0 \
        -sTEXTDECODER=0 \
        -sINCOMING_MODULE_JS_API=[] \
        -sEXPORT_NAME='createModule' \
        -sUSE_ES6_IMPORT_META=0")
        
    add_compile_options(-O3 -flto -fno-rtti -fno-exceptions)
else()
    find_package(SDL2 REQUIRED)
    add_compile_options(-O3)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

add_executable(aisnake_web src/main.cpp)

if(NOT EMSCRIPTEN)
    target_link_libraries(aisnake_web ${SDL2_LIBRARIES})
endif()

if(EMSCRIPTEN)
    set_target_properties(aisnake_web PROPERTIES
        OUTPUT_NAME "aisnake_web"
        SUFFIX ".html"
    )
endif()
