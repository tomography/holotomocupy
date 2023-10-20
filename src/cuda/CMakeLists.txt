enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
#find_package(NumPy REQUIRED)
#include_directories(${NumPy_INCLUDE_DIRS})

find_package(OpenMP)

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(tomo.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(holo.i PROPERTIES CPLUSPLUS ON)


swig_add_library(tomo LANGUAGE python SOURCES tomo.i tomo.cu)
swig_add_library(holo LANGUAGE python SOURCES holo.i holo.cu)


swig_link_libraries(tomo cudart cufft OpenMP::OpenMP_CXX)
swig_link_libraries(holo cudart cufft)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fopenmp")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tomo.py" DESTINATION holotomo)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/holo.py" DESTINATION holotomo)


install(TARGETS tomo holo LIBRARY DESTINATION holotomo)
