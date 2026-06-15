dir_abs_path=$(pwd)
rm -rf ${dir_abs_path}/yaml-cpp-install/
mkdir -p ${dir_abs_path}/yaml-cpp-install/
cd yaml-cpp-install/
rm -rf ./build/
mkdir build
cd build
cmake ../../yaml-cpp/ -DYAML_BUILD_SHARED_LIBS=ON \
                      -DCMAKE_INSTALL_PREFIX=${dir_abs_path}/yaml-cpp-install
make -j10
make install
cd ..
mv ./lib/cmake/yaml-cpp/yaml-cpp-config.cmake ./lib/cmake/yaml-cpp/YAML_CPPConfig.cmake 
