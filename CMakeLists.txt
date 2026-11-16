cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# The static libraries (RocketSim, RLGymCPP) get linked into the GigaLearnCPP shared library,
# so on Linux/macOS they must be compiled as position-independent code (-fPIC), otherwise the link fails
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	# In a shared library, every function that isn't inline can be replaced by another library when the program loads.
	# GCC therefore doesn't inline such functions, and calls them through a lookup table.
	# Nothing here relies on that. Without it the game simulation (RocketSim/Bullet) measured about 5% faster.
	add_compile_options(-fno-semantic-interposition)
	add_link_options(-Wl,-Bsymbolic-functions)

	# Math functions don't have to set errno, so sqrt() becomes a single instruction. The results are the same
	add_compile_options(-fno-math-errno)
endif()

# Default to an optimized build (the learner is extremely slow in debug builds)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)