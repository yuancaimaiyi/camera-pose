# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuancaimaiyi/桌面/cameraPose

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuancaimaiyi/桌面/cameraPose/build

# Include any dependencies generated for this target.
include CMakeFiles/campose.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/campose.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/campose.dir/flags.make

CMakeFiles/campose.dir/main.cpp.o: CMakeFiles/campose.dir/flags.make
CMakeFiles/campose.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/campose.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/campose.dir/main.cpp.o -c /home/yuancaimaiyi/桌面/cameraPose/main.cpp

CMakeFiles/campose.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/campose.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yuancaimaiyi/桌面/cameraPose/main.cpp > CMakeFiles/campose.dir/main.cpp.i

CMakeFiles/campose.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/campose.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yuancaimaiyi/桌面/cameraPose/main.cpp -o CMakeFiles/campose.dir/main.cpp.s

CMakeFiles/campose.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/campose.dir/main.cpp.o.requires

CMakeFiles/campose.dir/main.cpp.o.provides: CMakeFiles/campose.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/campose.dir/build.make CMakeFiles/campose.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/campose.dir/main.cpp.o.provides

CMakeFiles/campose.dir/main.cpp.o.provides.build: CMakeFiles/campose.dir/main.cpp.o


# Object files for target campose
campose_OBJECTS = \
"CMakeFiles/campose.dir/main.cpp.o"

# External object files for target campose
campose_EXTERNAL_OBJECTS =

campose: CMakeFiles/campose.dir/main.cpp.o
campose: CMakeFiles/campose.dir/build.make
campose: /usr/local/ceres/lib/libceres.a
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_img_hash.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_sfm.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: /usr/lib/x86_64-linux-gnu/libglog.so
campose: /usr/lib/x86_64-linux-gnu/libspqr.so
campose: /usr/lib/x86_64-linux-gnu/libtbb.so
campose: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
campose: /usr/lib/x86_64-linux-gnu/libcholmod.so
campose: /usr/lib/x86_64-linux-gnu/libccolamd.so
campose: /usr/lib/x86_64-linux-gnu/libcamd.so
campose: /usr/lib/x86_64-linux-gnu/libcolamd.so
campose: /usr/lib/x86_64-linux-gnu/libamd.so
campose: /usr/lib/liblapack.so
campose: /usr/lib/libf77blas.so
campose: /usr/lib/libatlas.so
campose: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
campose: /usr/lib/x86_64-linux-gnu/librt.so
campose: /usr/lib/liblapack.so
campose: /usr/lib/libf77blas.so
campose: /usr/lib/libatlas.so
campose: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
campose: /usr/lib/x86_64-linux-gnu/librt.so
campose: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
campose: CMakeFiles/campose.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable campose"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/campose.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/campose.dir/build: campose

.PHONY : CMakeFiles/campose.dir/build

CMakeFiles/campose.dir/requires: CMakeFiles/campose.dir/main.cpp.o.requires

.PHONY : CMakeFiles/campose.dir/requires

CMakeFiles/campose.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/campose.dir/cmake_clean.cmake
.PHONY : CMakeFiles/campose.dir/clean

CMakeFiles/campose.dir/depend:
	cd /home/yuancaimaiyi/桌面/cameraPose/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuancaimaiyi/桌面/cameraPose /home/yuancaimaiyi/桌面/cameraPose /home/yuancaimaiyi/桌面/cameraPose/build /home/yuancaimaiyi/桌面/cameraPose/build /home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles/campose.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/campose.dir/depend

