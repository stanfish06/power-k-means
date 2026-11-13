cmake_minimum_required(VERSION 3.23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(power-k-means LANGUAGES CXX)
cmake_policy(SET CMP0167 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Boost headers (uBLAS, format): use an installed Boost (conan, nix, system)
# when CMake can find its config package; otherwise download the pinned
# headers-only release archive. PKM_BOOST_URL accepts a local file path to
# build offline.
set(PKM_BOOST_URL
    "https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-b2-nodocs.tar.xz"
    CACHE STRING "Boost release archive used when no installed Boost is found")
set(PKM_BOOST_SHA256
    "9e6bee9ab529fb2b0733049692d57d10a72202af085e553539a05b4204211a6f"
    CACHE STRING "SHA-256 of PKM_BOOST_URL")
find_package(Boost 1.70 CONFIG QUIET)
if(Boost_FOUND)
  message(STATUS "Using installed Boost ${Boost_VERSION} at ${Boost_DIR}")
else()
  include(FetchContent)
  # SOURCE_SUBDIR points at a missing directory so the archive is only
  # unpacked, not added as a CMake subproject
  FetchContent_Declare(
    pkm_boost
    URL "${PKM_BOOST_URL}"
    URL_HASH "SHA256=${PKM_BOOST_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR do-not-configure)
  message(STATUS "Boost config not found; fetching ${PKM_BOOST_URL}")
  FetchContent_MakeAvailable(pkm_boost)
  add_library(Boost::headers INTERFACE IMPORTED)
  target_include_directories(Boost::headers SYSTEM
                             INTERFACE "${pkm_boost_SOURCE_DIR}")
endif()

add_subdirectory(src)

# scikit-build-core sets SKBUILD when building the Python package; a manual
# build turns the extension on with -DPKM_BUILD_PYTHON=ON
if(SKBUILD)
  set(PKM_BUILD_PYTHON_DEFAULT ON)
else()
  set(PKM_BUILD_PYTHON_DEFAULT OFF)
endif()
option(PKM_BUILD_PYTHON "Build the Cython extension module"
       ${PKM_BUILD_PYTHON_DEFAULT})
if(PKM_BUILD_PYTHON)
  add_subdirectory(python)
endif()
