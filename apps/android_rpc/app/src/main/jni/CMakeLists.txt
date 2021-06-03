cmake_minimum_required(VERSION 3.4.1)
# whether enable OpenCL during compile
#set(USE_OPENCL FALSE)
## whether to enable Vulkan during compile
#set(USE_VULKAN FALSE)
## whether to eanble contrib random functions during compile
#set(USE_RANDOM TRUE)
## the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
#set(ADD_LDLIBS OFF)
## the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
#set(ADD_C_INCLUDES OFF)
##APP_PLATFORM = android-24
#set(ROOT_PATH ./../../../../../..)
#
#if(${USE_VULKAN})
#    if(${CMAKE_ANDROID_API} GREATER_EQUAL 24)
#    message("android版本太低，没有VULKAN")
#    endif()
#endif()
#
#include_directories(${ROOT_PATH}/include)
#include_directories(${ROOT_PATH}/3rdparty/dlpack/include)
#include_directories(${ROOT_PATH}/3rdparty/dmlc-core/include)
#
#
#
#add_library(
#        tvm4j_runtime_packed
#        shared
#        org_apache_tvm_native_c_api.cc
#)
add_library( # Sets the name of the library.
        tvm4j_runtime_packed

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).


        IMPORTED

        )
message(${CMAKE_SOURCE_DIR}/../libs/arm64-v8a/libtvm4j_runtime_packed.so)
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set_target_properties(
        tvm4j_runtime_packed

            PROPERTIES IMPORTED_LOCATION

        ${CMAKE_SOURCE_DIR}/../libs/arm64-v8a/libtvm4j_runtime_packed.so

)
elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set_target_properties(
            tvm4j_runtime_packed

            PROPERTIES IMPORTED_LOCATION

            ${CMAKE_SOURCE_DIR}/../libs/armeabi-v7a/libtvm4j_runtime_packed.so

    )
endif()