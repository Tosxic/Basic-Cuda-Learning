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
CMAKE_SOURCE_DIR = "/home/tosxic/workspace/cuda/08 block shared memory"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/tosxic/workspace/cuda/08 block shared memory/build"

# Include any dependencies generated for this target.
include CMakeFiles/sum_opt5.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sum_opt5.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sum_opt5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sum_opt5.dir/flags.make

CMakeFiles/sum_opt5.dir/sum_opt5.cu.o: CMakeFiles/sum_opt5.dir/flags.make
CMakeFiles/sum_opt5.dir/sum_opt5.cu.o: ../sum_opt5.cu
CMakeFiles/sum_opt5.dir/sum_opt5.cu.o: CMakeFiles/sum_opt5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/tosxic/workspace/cuda/08 block shared memory/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sum_opt5.dir/sum_opt5.cu.o"
	/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/sum_opt5.dir/sum_opt5.cu.o -MF CMakeFiles/sum_opt5.dir/sum_opt5.cu.o.d -x cu -c "/home/tosxic/workspace/cuda/08 block shared memory/sum_opt5.cu" -o CMakeFiles/sum_opt5.dir/sum_opt5.cu.o

CMakeFiles/sum_opt5.dir/sum_opt5.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sum_opt5.dir/sum_opt5.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sum_opt5.dir/sum_opt5.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sum_opt5.dir/sum_opt5.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sum_opt5
sum_opt5_OBJECTS = \
"CMakeFiles/sum_opt5.dir/sum_opt5.cu.o"

# External object files for target sum_opt5
sum_opt5_EXTERNAL_OBJECTS =

../bin/sum_opt5: CMakeFiles/sum_opt5.dir/sum_opt5.cu.o
../bin/sum_opt5: CMakeFiles/sum_opt5.dir/build.make
../bin/sum_opt5: CMakeFiles/sum_opt5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/tosxic/workspace/cuda/08 block shared memory/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ../bin/sum_opt5"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sum_opt5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sum_opt5.dir/build: ../bin/sum_opt5
.PHONY : CMakeFiles/sum_opt5.dir/build

CMakeFiles/sum_opt5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sum_opt5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sum_opt5.dir/clean

CMakeFiles/sum_opt5.dir/depend:
	cd "/home/tosxic/workspace/cuda/08 block shared memory/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/tosxic/workspace/cuda/08 block shared memory" "/home/tosxic/workspace/cuda/08 block shared memory" "/home/tosxic/workspace/cuda/08 block shared memory/build" "/home/tosxic/workspace/cuda/08 block shared memory/build" "/home/tosxic/workspace/cuda/08 block shared memory/build/CMakeFiles/sum_opt5.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/sum_opt5.dir/depend

