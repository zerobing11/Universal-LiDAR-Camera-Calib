echo "Start Building ThirdParty"
cd src/ThirdParty/
rm -rf build
mkdir build
cd build 
cmake ..
make
