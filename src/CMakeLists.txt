add_library(power_kmeans_core STATIC power_kmeans.cpp)
target_include_directories(power_kmeans_core
                           PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(power_kmeans_core PUBLIC Boost::headers)
# PIC so the static library can be linked into the Python extension module
set_target_properties(power_kmeans_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# the demo is not part of the wheel
if(NOT SKBUILD)
  add_executable(main main.cpp)
  target_link_libraries(main PRIVATE power_kmeans_core)
endif()
