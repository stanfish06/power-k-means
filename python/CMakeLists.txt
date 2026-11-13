find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# Cython comes from the build requirements in pyproject.toml
set(PKM_PYX "${CMAKE_CURRENT_SOURCE_DIR}/power_kmeans/_core.pyx")
set(PKM_PYX_CPP "${CMAKE_CURRENT_BINARY_DIR}/_core.cpp")
add_custom_command(
  OUTPUT "${PKM_PYX_CPP}"
  COMMAND Python::Interpreter -m cython --cplus -3 "${PKM_PYX}" -o
          "${PKM_PYX_CPP}"
  DEPENDS "${PKM_PYX}"
  COMMENT "Cythonizing _core.pyx"
  VERBATIM)

python_add_library(_core MODULE WITH_SOABI "${PKM_PYX_CPP}")
target_link_libraries(_core PRIVATE power_kmeans_core)
install(TARGETS _core DESTINATION power_kmeans)

# Manual builds: also install the Python files, so that
# `cmake --install <build> --prefix <stage>` gives an importable package under
# <stage>. Wheels get them from wheel.packages in pyproject.toml.
if(NOT SKBUILD)
  install(
    DIRECTORY power_kmeans/
    DESTINATION power_kmeans
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "py.typed")
endif()
