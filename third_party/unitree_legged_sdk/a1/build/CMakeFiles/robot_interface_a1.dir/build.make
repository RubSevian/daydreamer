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
CMAKE_SOURCE_DIR = /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build

# Include any dependencies generated for this target.
include CMakeFiles/robot_interface_a1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/robot_interface_a1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/robot_interface_a1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/robot_interface_a1.dir/flags.make

CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o: CMakeFiles/robot_interface_a1.dir/flags.make
CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o: ../python_interface.cpp
CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o: CMakeFiles/robot_interface_a1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o -MF CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o.d -o CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o -c /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/python_interface.cpp

CMakeFiles/robot_interface_a1.dir/python_interface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot_interface_a1.dir/python_interface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/python_interface.cpp > CMakeFiles/robot_interface_a1.dir/python_interface.cpp.i

CMakeFiles/robot_interface_a1.dir/python_interface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot_interface_a1.dir/python_interface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/python_interface.cpp -o CMakeFiles/robot_interface_a1.dir/python_interface.cpp.s

# Object files for target robot_interface_a1
robot_interface_a1_OBJECTS = \
"CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o"

# External object files for target robot_interface_a1
robot_interface_a1_EXTERNAL_OBJECTS =

robot_interface_a1.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface_a1.dir/python_interface.cpp.o
robot_interface_a1.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface_a1.dir/build.make
robot_interface_a1.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface_a1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module robot_interface_a1.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/robot_interface_a1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/robot_interface_a1.dir/build: robot_interface_a1.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/robot_interface_a1.dir/build

CMakeFiles/robot_interface_a1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/robot_interface_a1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/robot_interface_a1.dir/clean

CMakeFiles/robot_interface_a1.dir/depend:
	cd /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1 /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1 /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/a1/build/CMakeFiles/robot_interface_a1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/robot_interface_a1.dir/depend

