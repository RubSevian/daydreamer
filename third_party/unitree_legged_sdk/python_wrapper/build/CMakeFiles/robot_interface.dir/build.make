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
CMAKE_SOURCE_DIR = /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build

# Include any dependencies generated for this target.
include CMakeFiles/robot_interface.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/robot_interface.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/robot_interface.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/robot_interface.dir/flags.make

CMakeFiles/robot_interface.dir/python_interface.cpp.o: CMakeFiles/robot_interface.dir/flags.make
CMakeFiles/robot_interface.dir/python_interface.cpp.o: ../python_interface.cpp
CMakeFiles/robot_interface.dir/python_interface.cpp.o: CMakeFiles/robot_interface.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/robot_interface.dir/python_interface.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/robot_interface.dir/python_interface.cpp.o -MF CMakeFiles/robot_interface.dir/python_interface.cpp.o.d -o CMakeFiles/robot_interface.dir/python_interface.cpp.o -c /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/python_interface.cpp

CMakeFiles/robot_interface.dir/python_interface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot_interface.dir/python_interface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/python_interface.cpp > CMakeFiles/robot_interface.dir/python_interface.cpp.i

CMakeFiles/robot_interface.dir/python_interface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot_interface.dir/python_interface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/python_interface.cpp -o CMakeFiles/robot_interface.dir/python_interface.cpp.s

# Object files for target robot_interface
robot_interface_OBJECTS = \
"CMakeFiles/robot_interface.dir/python_interface.cpp.o"

# External object files for target robot_interface
robot_interface_EXTERNAL_OBJECTS =

/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface.dir/python_interface.cpp.o
/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface.dir/build.make
/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so: CMakeFiles/robot_interface.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/robot_interface.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/robot_interface.dir/build: /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/robot_interface.dir/build

CMakeFiles/robot_interface.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/robot_interface.dir/cmake_clean.cmake
.PHONY : CMakeFiles/robot_interface.dir/clean

CMakeFiles/robot_interface.dir/depend:
	cd /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build /home/sber/Documents/daydreamer/daydreamer/third_party/unitree_legged_sdk/python_wrapper/build/CMakeFiles/robot_interface.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/robot_interface.dir/depend

