@echo off
setlocal
pushd "%~dp0.."
if not exist build mkdir build
cd build
cmake -S .. -B . -DBUILD_TESTS=ON
cmake --build . --config Release --target engine_tests
ctest -C Release --output-on-failure
popd
endlocal
