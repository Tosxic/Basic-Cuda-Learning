# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/tosxic/workspace/cuda/05 math"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/tosxic/workspace/cuda/05 math/build"

# Include any dependencies generated for this target.
include CMakeFiles/saxpy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/saxpy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/saxpy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/saxpy.dir/flags.make

CMakeFiles/saxpy.dir/saxpy.cu.o: CMakeFiles/saxpy.dir/flags.make
CMakeFiles/saxpy.dir/saxpy.cu.o: ../saxpy.cu
CMakeFiles/saxpy.dir/saxpy.cu.o: CMakeFiles/saxpy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/tosxic/workspace/cuda/05 math/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/saxpy.dir/saxpy.cu.o"
	/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/saxpy.dir/saxpy.cu.o -MF CMakeFiles/saxpy.dir/saxpy.cu.o.d -x cu -c "/home/tosxic/workspace/cuda/05 math/saxpy.cu" -o CMakeFiles/saxpy.dir/saxpy.cu.o

CMakeFiles/saxpy.dir/saxpy.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/saxpy.dir/saxpy.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/saxpy.dir/saxpy.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/saxpy.dir/saxpy.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target saxpy
saxpy_OBJECTS = \
"CMakeFiles/saxpy.dir/saxpy.cu.o"

# External object files for target saxpy
saxpy_EXTERNAL_OBJECTS =

../bin/saxpy: CMakeFiles/saxpy.dir/saxpy.cu.o
../bin/saxpy: CMakeFiles/saxpy.dir/build.make
../bin/saxpy: CMakeFiles/saxpy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/tosxic/workspace/cuda/05 math/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ../bin/saxpy"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/saxpy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/saxpy.dir/build: ../bin/saxpy
.PHONY : CMakeFiles/saxpy.dir/build

CMakeFiles/saxpy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/saxpy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/saxpy.dir/clean

CMakeFiles/saxpy.dir/depend:
	cd "/home/tosxic/workspace/cuda/05 math/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/tosxic/workspace/cuda/05 math" "/home/tosxic/workspace/cuda/05 math" "/home/tosxic/workspace/cuda/05 math/build" "/home/tosxic/workspace/cuda/05 math/build" "/home/tosxic/workspace/cuda/05 math/build/CMakeFiles/saxpy.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/saxpy.dir/depend

