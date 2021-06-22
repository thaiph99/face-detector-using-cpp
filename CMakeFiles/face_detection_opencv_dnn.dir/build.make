# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /snap/cmake/882/bin/cmake

# The command to remove a file.
RM = /snap/cmake/882/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/it/Data/thaiphData/face-detector-using-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/it/Data/thaiphData/face-detector-using-cpp

# Include any dependencies generated for this target.
include CMakeFiles/face_detection_opencv_dnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/face_detection_opencv_dnn.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/face_detection_opencv_dnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/face_detection_opencv_dnn.dir/flags.make

CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o: CMakeFiles/face_detection_opencv_dnn.dir/flags.make
CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o: face_detection_opencv_dnn.cpp
CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o: CMakeFiles/face_detection_opencv_dnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/it/Data/thaiphData/face-detector-using-cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o -MF CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o.d -o CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o -c /home/it/Data/thaiphData/face-detector-using-cpp/face_detection_opencv_dnn.cpp

CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/it/Data/thaiphData/face-detector-using-cpp/face_detection_opencv_dnn.cpp > CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.i

CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/it/Data/thaiphData/face-detector-using-cpp/face_detection_opencv_dnn.cpp -o CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.s

# Object files for target face_detection_opencv_dnn
face_detection_opencv_dnn_OBJECTS = \
"CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o"

# External object files for target face_detection_opencv_dnn
face_detection_opencv_dnn_EXTERNAL_OBJECTS =

face_detection_opencv_dnn: CMakeFiles/face_detection_opencv_dnn.dir/face_detection_opencv_dnn.cpp.o
face_detection_opencv_dnn: CMakeFiles/face_detection_opencv_dnn.dir/build.make
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
face_detection_opencv_dnn: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
face_detection_opencv_dnn: CMakeFiles/face_detection_opencv_dnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/it/Data/thaiphData/face-detector-using-cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable face_detection_opencv_dnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/face_detection_opencv_dnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/face_detection_opencv_dnn.dir/build: face_detection_opencv_dnn
.PHONY : CMakeFiles/face_detection_opencv_dnn.dir/build

CMakeFiles/face_detection_opencv_dnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/face_detection_opencv_dnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/face_detection_opencv_dnn.dir/clean

CMakeFiles/face_detection_opencv_dnn.dir/depend:
	cd /home/it/Data/thaiphData/face-detector-using-cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/it/Data/thaiphData/face-detector-using-cpp /home/it/Data/thaiphData/face-detector-using-cpp /home/it/Data/thaiphData/face-detector-using-cpp /home/it/Data/thaiphData/face-detector-using-cpp /home/it/Data/thaiphData/face-detector-using-cpp/CMakeFiles/face_detection_opencv_dnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/face_detection_opencv_dnn.dir/depend

