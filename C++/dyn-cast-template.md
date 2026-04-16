# C++ dyn_cast 模板详解

## 🧩 一、背景：C++ 的 dynamic_cast

标准 C++ 的 RTTI（运行时类型识别）提供 dynamic_cast：

```cpp
Base* b = new Derived();
if (Derived* d = dynamic_cast<Derived*>(b)) {
    d->foo();
}
```

- 会在运行时检查对象的真实类型；
- 若 b 真指向 Derived，转换成功；
- 否则返回 nullptr；
- 缺点：需要启用 RTTI（Run-Time Type Information），性能有额外开销；
- 在编译器基础设施（如 LLVM、MLIR）中通常禁用 RTTI，因此不能使用标准 dynamic_cast。

## ⚙️ 二、LLVM/MLIR 的替代方案：dyn_cast 模板

LLVM 自己实现了一套模板机制，实现了类似 dynamic_cast 的功能，但完全在编译期静态分派 + 轻量运行时类型检查完成。

核心接口位于：
```cpp
#include "llvm/Support/Casting.h"
```

或在 MLIR 中：
```cpp
#include "mlir/Support/LLVM.h"
```

## 🧠 三、isa / cast / dyn_cast 三兄弟

这三者是配套使用的模板函数：

| 函数 | 作用 |
|------|------|
| `isa<T>(ptr)` | 判断对象是否是类型 T |
| `cast<T>(ptr)` | 强制转换（假定成功） |
| `dyn_cast<T>(ptr)` | 动态安全转换（失败返回 nullptr） |

### ✅ 例子
```cpp
#include "llvm/Support/Casting.h"
#include <iostream>
using namespace llvm;

struct Base {
    enum Kind { K_Base, K_Derived };
    Kind kind;
    Base(Kind k) : kind(k) {}
    Kind getKind() const { return kind; }
};

struct Derived : Base {
    Derived() : Base(K_Derived) {}
    static bool classof(const Base* b) {
        return b->getKind() == K_Derived;
    }
};

int main() {
    Base* b = new Derived();

    if (isa<Derived>(b)) { // 判断类型
        Derived* d = dyn_cast<Derived>(b); // 安全转换
        std::cout << "b is Derived\n";
    }
}
```

输出：
```
b is Derived
```

## ⚙️ 四、工作原理详解

dyn_cast 实际上是基于 isa 的封装：

```cpp
template <typename To, typename From>
To* dyn_cast(From* f) {
    return isa<To>(f) ? static_cast<To*>(f) : nullptr;
}
```

而 isa 的判断依赖被转换类型 To 提供的静态方法：
```cpp
static bool classof(const BaseType* from);
```

这样可以在不同层级中自定义类型判断逻辑。
即：类自己知道如何识别自己。

## 🧩 五、在 MLIR 中的使用

在 MLIR IR 系统中，dyn_cast 被广泛用于：

- **Operation 类型判断**
  ```cpp
  if (auto addOp = dyn_cast<AddIOp>(op)) {
      // op 是加法操作
  }
  ```

- **Type 判断**
  ```cpp
  if (auto tensorType = dyn_cast<TensorType>(value.getType())) {
      ...
  }
  ```

- **Attribute 判断**
  ```cpp
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      ...
  }
  ```

它们底层都要求类定义：
```cpp
static bool classof(SomeBaseType *);
```

例如在 MLIR 中：
```cpp
static bool classof(Operation *op) {
    return op->getName() == "arith.addi";
}
```

## 🧮 六、三者区别总结表

| 函数 | 功能 | 返回 | 失败行为 | 用法场景 |
|------|------|------|----------|----------|
| `isa<T>(x)` | 判断是否是类型 T | bool | 无 | if 判断 |
| `cast<T>(x)` | 静态转换 | T* | 断言失败（崩溃） | 你确定类型正确时 |
| `dyn_cast<T>(x)` | 动态安全转换 | T* | 返回 nullptr | 不确定类型时 |

## ⚡ 七、为什么比 dynamic_cast 高效

- 不使用 RTTI（编译期结构体内部实现）；
- 基于静态模板展开 + 自定义 classof() 检查；
- 无虚表指针查找；
- 可用于编译器 IR 的层级体系（Operation, Type, Attribute）；
- 允许定制更细粒度的类型判定逻辑。