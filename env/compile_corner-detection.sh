dir_abs_path=$(pwd)
rm -rf ${dir_abs_path}/corner-detection-install/
mkdir -p ${dir_abs_path}/corner-detection-install/
cd corner-detection-install/
rm -rf ./build/
mkdir build
cd build
cmake ../../corner-detection/ -DCMAKE_INSTALL_PREFIX=${dir_abs_path}/corner-detection-install
make -j10
make install
cd ..

