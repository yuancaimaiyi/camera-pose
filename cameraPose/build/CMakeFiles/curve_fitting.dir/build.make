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
include CMakeFiles/curve_fitting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/curve_fitting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/curve_fitting.dir/flags.make

CMakeFiles/curve_fitting.dir/main.cpp.o: CMakeFiles/curve_fitting.dir/flags.make
CMakeFiles/curve_fitting.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/curve_fitting.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/curve_fitting.dir/main.cpp.o -c /home/yuancaimaiyi/桌面/cameraPose/main.cpp

CMakeFiles/curve_fitting.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/curve_fitting.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yuancaimaiyi/桌面/cameraPose/main.cpp > CMakeFiles/curve_fitting.dir/main.cpp.i

CMakeFiles/curve_fitting.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/curve_fitting.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yuancaimaiyi/桌面/cameraPose/main.cpp -o CMakeFiles/curve_fitting.dir/main.cpp.s

CMakeFiles/curve_fitting.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/curve_fitting.dir/main.cpp.o.requires

CMakeFiles/curve_fitting.dir/main.cpp.o.provides: CMakeFiles/curve_fitting.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/curve_fitting.dir/build.make CMakeFiles/curve_fitting.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/curve_fitting.dir/main.cpp.o.provides

CMakeFiles/curve_fitting.dir/main.cpp.o.provides.build: CMakeFiles/curve_fitting.dir/main.cpp.o


# Object files for target curve_fitting
curve_fitting_OBJECTS = \
"CMakeFiles/curve_fitting.dir/main.cpp.o"

# External object files for target curve_fitting
curve_fitting_EXTERNAL_OBJECTS =

curve_fitting: CMakeFiles/curve_fitting.dir/main.cpp.o
curve_fitting: CMakeFiles/curve_fitting.dir/build.make
curve_fitting: /usr/local/ceres/lib/libceres.a
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_img_hash.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_sfm.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: /usr/lib/x86_64-linux-gnu/libglog.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libspqr.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libtbb.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libcamd.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libamd.so
curve_fitting: /usr/lib/liblapack.so
curve_fitting: /usr/lib/libf77blas.so
curve_fitting: /usr/lib/libatlas.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
curve_fitting: /usr/lib/x86_64-linux-gnu/librt.so
curve_fitting: /usr/lib/liblapack.so
curve_fitting: /usr/lib/libf77blas.so
curve_fitting: /usr/lib/libatlas.so
curve_fitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
curve_fitting: /usr/lib/x86_64-linux-gnu/librt.so
curve_fitting: /usr/local/opencv3.4.3/lib/libopencv_world.so.3.4.3
curve_fitting: CMakeFiles/curve_fitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable curve_fitting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/curve_fitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/curve_fitting.dir/build: curve_fitting

.PHONY : CMakeFiles/curve_fitting.dir/build

CMakeFiles/curve_fitting.dir/requires: CMakeFiles/curve_fitting.dir/main.cpp.o.requires

.PHONY : CMakeFiles/curve_fitting.dir/requires

CMakeFiles/curve_fitting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/curve_fitting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/curve_fitting.dir/clean

CMakeFiles/curve_fitting.dir/depend:
	cd /home/yuancaimaiyi/桌面/cameraPose/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuancaimaiyi/桌面/cameraPose /home/yuancaimaiyi/桌面/cameraPose /home/yuancaimaiyi/桌面/cameraPose/build /home/yuancaimaiyi/桌面/cameraPose/build /home/yuancaimaiyi/桌面/cameraPose/build/CMakeFiles/curve_fitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/curve_fitting.dir/depend

