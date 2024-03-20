# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

# TODO: A copy also exists in nodos repo. Create a submodule or copy to Tools.

function(generate_flatbuffers FLATC_PATH FBS_FOLDERS OUT_FOLDER TEMP_FOLDER OUT_LANGUAGE INCLUDE_FOLDERS)
    # Check if flatbuffers compiler is available
    message("Looking for flatbuffers compiler at ${FLATC_PATH}")
    find_program(FLATC "${FLATC_PATH}")
    # Check if flatbuffers compiler is available (cross platform
    if (NOT FLATC)
        message(FATAL_ERROR "Flatbuffers compiler not found. Please set FLATC_PATH variable.")
    endif()
    file(GLOB_RECURSE FBS_FILES ${FBS_FOLDERS}/*.fbs)
    message(STATUS "Using flatbuffers compiler: " ${FLATC})
    file(REMOVE_RECURSE ${TEMP_FOLDER})
    file(MAKE_DIRECTORY ${TEMP_FOLDER})
    foreach(FBS_FILE ${FBS_FILES})
        get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME_WE)
        set(FBS_OUT_HEADER "${FBS_FILE_NAME}_generated.h")
        set(FBS_OUT_GRPC_HEADER "${FBS_FILE_NAME}.grpc.fb.h")
        set(FBS_OUT_GRPC_SRC "${FBS_FILE_NAME}.grpc.fb.cc")
        message(STATUS "Generating: ${FBS_FILE}")
        set(INCLUDE_PARAMS "")
        foreach(INCLUDE ${INCLUDE_FOLDERS})
            set(INCLUDE_PARAMS ${INCLUDE_PARAMS} -I ${INCLUDE})
        endforeach()
        execute_process(COMMAND ${FLATC}
                                    -o ${TEMP_FOLDER}
                                    ${INCLUDE_PARAMS}
                                    ${FBS_FILE}
                                    --${OUT_LANGUAGE}
                                    --grpc ${FBS_FILE} 
                                    --gen-mutable
                                    --gen-name-strings
                                    --gen-object-api
                                    --gen-compare
                                    --cpp-std=c++17
                                    --cpp-static-reflection
                                    --scoped-enums
                                    --unknown-json
                                    --reflect-types
                                    --reflect-names
                                    --cpp-include array
                                    # --force-empty-vectors
                                    # --force-empty
                                    # --force-defaults
                                    --object-prefix "T"
                                    --object-suffix ""
                                    RESULT_VARIABLE FLATBUFFERSC_RESULT)
        # Check return code 
        if (NOT ${FLATBUFFERSC_RESULT} EQUAL "0")
            message(FATAL_ERROR "Failed to compile flatbuffers files. Process returned ${FLATBUFFERSC_RESULT}.")
        endif()
        source_group("FlatBuffers Files" FILES ${FBS_FILE})
    endforeach()
    # Compare folders ${TEMP_FOLDER} and ${OUT_FOLDER} and copy only changed or new files
    file(GLOB_RECURSE TEMP_FILES ${TEMP_FOLDER}/*)
    foreach(TEMP_FILE ${TEMP_FILES})
        get_filename_component(TEMP_FILE_NAME ${TEMP_FILE} NAME)
        set(OUT_FILE "${OUT_FOLDER}/${TEMP_FILE_NAME}")
        if (NOT EXISTS ${OUT_FILE})
            message(STATUS "Copying ${TEMP_FILE} to ${OUT_FILE}")
            file(COPY ${TEMP_FILE} DESTINATION ${OUT_FOLDER})
        else()
            file(MD5 ${TEMP_FILE} TEMP_MD5)
            file(MD5 ${OUT_FILE} OUT_MD5)
            if (NOT ${TEMP_MD5} STREQUAL ${OUT_MD5})
                message(STATUS "Copying ${TEMP_FILE} to ${OUT_FILE}")
                file(COPY ${TEMP_FILE} DESTINATION ${OUT_FOLDER})
            endif()
        endif()
    endforeach()
    # Remove temporary folder
    file(REMOVE_RECURSE ${TEMP_FOLDER})
endfunction()