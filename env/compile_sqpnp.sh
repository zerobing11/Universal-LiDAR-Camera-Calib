dir_abs_path=$(pwd)
rm -rf ${dir_abs_path}/sqpnp-install/
mkdir -p ${dir_abs_path}/sqpnp-install/
cd sqpnp-install/
rm -rf ./build/
mkdir build
cd build
cmake ../../sqpnp/ -DCMAKE_INSTALL_PREFIX=${dir_abs_path}/sqpnp-install
make -j10
make install
cd ..

