here=File.dirname(__FILE__)
$slu_home=File.expand_path("#{here}")
require "#{$slu_home}/rakefile.rb"

$CFLAGS = "-fpermissive -c -pipe -Wall -W   -DQT_SHARED -DQT_NO_DEBUG -DQT_THREAD_SUPPORT "

desc "Build everything."
lib = "c/libspatial_features.so"
swig_lib = "swig/_spatial_features_cxx.so"
ldflags = "-lgsl_utilities"



make_o_targets(:build_c, FileList["c/**/*.c"], ".c", "-Wall")
make_copy_targets(:build_c, FileList["c/**/*.h"], $include_build_dir)
make_lib_targets(:build_c, lib, FileList["c/*.c"], ldflags, $lib_build_dir)

make_swig_targets(["swig/spatial_features_cxx.i"], 
                  FileList["c/*.h"],
                  swig_lib, "-Ic -I/usr/include/gsl", 
                  ldflags + " -lspatial_features")

make_swig_targets(["swig/gsl_utilities.i"], 
                  FileList["c/*.h"],
                  "swig/_gsl_utilities.so", 
                  "-Ic", "-lgsl_utilities")



task :all => [:build_c, :build_swig, :build_python]
make_python_targets(:build_python, FileList["python/**/*.py"])

task :clean do
  sh "rm -f c/*.o"
  sh "rm -f c/*.so"
  sh "rm -f swig/*.o"
  sh "rm -f swig/*.so"
  sh "rm -f swig/*.cxx"
  sh "rm -f swig/*.py"
  sh "rm -f swig/*.h"
end



desc "Run the test cases."
python_task :partial_tests  do
  #python("/usr/bin/nosetests -v -s #{$pbd}/spatial_features/test/cmath2d_test.py")
  #python("/usr/bin/nosetests -v -s #{$pbd}/spatial_features/test/math3d_test.py")
  #python("/usr/bin/nosetests -v -s #{$pbd}/spatial_features/test/feature_map_test.py")
  python("/usr/bin/nosetests -v -s #{$pbd}/spatial_features/test/compare_python_and_c_test.py")
  #sh("/usr/bin/nosetests -v -s `find python -name '*_test.py'`")

end
