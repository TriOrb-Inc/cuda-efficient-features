#!bin/bash

ls build >> /dev/null 2>&1
if [ $? -ne 0 ]; then
    mkdir build
fi
cd build
cmake -DBUILD_TESTS=ON ../ &&\
make &&\
./tests/tests
#./tests/tests
#gdb -ex "set print thread-events off" -ex run --args ./tests/tests
