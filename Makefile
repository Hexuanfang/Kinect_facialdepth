# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ajith92/Kinect/Xbox_tracker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ajith92/Kinect/Xbox_tracker

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ajith92/Kinect/Xbox_tracker/CMakeFiles /home/ajith92/Kinect/Xbox_tracker/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ajith92/Kinect/Xbox_tracker/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named listener

# Build rule for target.
listener: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 listener
.PHONY : listener

# fast build rule for target.
listener/fast:
	$(MAKE) -f CMakeFiles/listener.dir/build.make CMakeFiles/listener.dir/build
.PHONY : listener/fast

#=============================================================================
# Target rules for targets named ros_listener

# Build rule for target.
ros_listener: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ros_listener
.PHONY : ros_listener

# fast build rule for target.
ros_listener/fast:
	$(MAKE) -f CMakeFiles/ros_listener.dir/build.make CMakeFiles/ros_listener.dir/build
.PHONY : ros_listener/fast

#=============================================================================
# Target rules for targets named xbox_tracker

# Build rule for target.
xbox_tracker: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 xbox_tracker
.PHONY : xbox_tracker

# fast build rule for target.
xbox_tracker/fast:
	$(MAKE) -f CMakeFiles/xbox_tracker.dir/build.make CMakeFiles/xbox_tracker.dir/build
.PHONY : xbox_tracker/fast

ros_listener.o: ros_listener.cpp.o
.PHONY : ros_listener.o

# target to build an object file
ros_listener.cpp.o:
	$(MAKE) -f CMakeFiles/ros_listener.dir/build.make CMakeFiles/ros_listener.dir/ros_listener.cpp.o
.PHONY : ros_listener.cpp.o

ros_listener.i: ros_listener.cpp.i
.PHONY : ros_listener.i

# target to preprocess a source file
ros_listener.cpp.i:
	$(MAKE) -f CMakeFiles/ros_listener.dir/build.make CMakeFiles/ros_listener.dir/ros_listener.cpp.i
.PHONY : ros_listener.cpp.i

ros_listener.s: ros_listener.cpp.s
.PHONY : ros_listener.s

# target to generate assembly for a file
ros_listener.cpp.s:
	$(MAKE) -f CMakeFiles/ros_listener.dir/build.make CMakeFiles/ros_listener.dir/ros_listener.cpp.s
.PHONY : ros_listener.cpp.s

xbox_listener.o: xbox_listener.cpp.o
.PHONY : xbox_listener.o

# target to build an object file
xbox_listener.cpp.o:
	$(MAKE) -f CMakeFiles/listener.dir/build.make CMakeFiles/listener.dir/xbox_listener.cpp.o
.PHONY : xbox_listener.cpp.o

xbox_listener.i: xbox_listener.cpp.i
.PHONY : xbox_listener.i

# target to preprocess a source file
xbox_listener.cpp.i:
	$(MAKE) -f CMakeFiles/listener.dir/build.make CMakeFiles/listener.dir/xbox_listener.cpp.i
.PHONY : xbox_listener.cpp.i

xbox_listener.s: xbox_listener.cpp.s
.PHONY : xbox_listener.s

# target to generate assembly for a file
xbox_listener.cpp.s:
	$(MAKE) -f CMakeFiles/listener.dir/build.make CMakeFiles/listener.dir/xbox_listener.cpp.s
.PHONY : xbox_listener.cpp.s

xbox_tracker.o: xbox_tracker.cpp.o
.PHONY : xbox_tracker.o

# target to build an object file
xbox_tracker.cpp.o:
	$(MAKE) -f CMakeFiles/xbox_tracker.dir/build.make CMakeFiles/xbox_tracker.dir/xbox_tracker.cpp.o
.PHONY : xbox_tracker.cpp.o

xbox_tracker.i: xbox_tracker.cpp.i
.PHONY : xbox_tracker.i

# target to preprocess a source file
xbox_tracker.cpp.i:
	$(MAKE) -f CMakeFiles/xbox_tracker.dir/build.make CMakeFiles/xbox_tracker.dir/xbox_tracker.cpp.i
.PHONY : xbox_tracker.cpp.i

xbox_tracker.s: xbox_tracker.cpp.s
.PHONY : xbox_tracker.s

# target to generate assembly for a file
xbox_tracker.cpp.s:
	$(MAKE) -f CMakeFiles/xbox_tracker.dir/build.make CMakeFiles/xbox_tracker.dir/xbox_tracker.cpp.s
.PHONY : xbox_tracker.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... listener"
	@echo "... rebuild_cache"
	@echo "... ros_listener"
	@echo "... xbox_tracker"
	@echo "... ros_listener.o"
	@echo "... ros_listener.i"
	@echo "... ros_listener.s"
	@echo "... xbox_listener.o"
	@echo "... xbox_listener.i"
	@echo "... xbox_listener.s"
	@echo "... xbox_tracker.o"
	@echo "... xbox_tracker.i"
	@echo "... xbox_tracker.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
