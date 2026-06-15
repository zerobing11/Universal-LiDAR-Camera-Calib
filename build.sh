if [ $# -eq 0 ]; then
	mkdir -p build
	cd build

elif [ "$1" = "clear" ]; then
	rm -rf build/
	mkdir -p build
	cd build
fi

cmake ..
make -j16>&1 | tee build.log
cd ..
