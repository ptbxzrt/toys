一、本项目使用现代C++实现了一些玩具

- [std::variant](https://github.com/ptbxzrt/toys/tree/main/variant)
- [生产者消费者模型](https://github.com/ptbxzrt/toys/tree/main/producer_consumer)

<!-- 二、安装第三方依赖
- 安装yalantinglibs。
```shell
## 在一个单独的路径下克隆仓库
git clone https://github.com/alibaba/yalantinglibs.git

## 编译，跳过一些不必要的样例编译和测试程序
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build .
## 安装，将yalantinglibs安装到指定路径下
cmake --install . --prefix {路径}/toys/3rd
``` -->

二、编译toys
```shell
git clone git@github.com:ptbxzrt/toys.git

## 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build . -j16
```

三、引用的第三方库
- yalantinglibs