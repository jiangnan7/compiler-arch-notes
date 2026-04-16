# C++ 四种 Cast 转换的区别与应用场景

在现代 C++ 开发（尤其是涉及到复杂编译器前端或 LLVM/MLIR 源码阅读时），类型转换的精确使用至关重要。

## 1. static_cast

**应用：** 用于良性转换，如基本数据类型转换（float 转 int），以及明确已知安全的类层次结构中的向上转换（子转父）或向下转换（父转子）。

**特点：** 编译时进行类型检查，不包含运行时的类型检查（RTTI）。如果向下转换的对象实际类型不符，会导致未定义行为。

**示例：**
```cpp
// 基本类型转换
float f = 3.14;
int i = static_cast<int>(f);  // 3

// 类层次结构转换
class Base {};
class Derived : public Base {};

Derived d;
Base* b = static_cast<Base*>(&d);  // 向上转换（安全）

Base* b2 = new Derived();
Derived* d2 = static_cast<Derived*>(b2);  // 向下转换（如果b2实际指向Derived，则安全）
```

## 2. dynamic_cast

**应用：** 主要用于多态类（带有虚函数）的安全向下转换。

**特点：** 在运行时检查转换的安全性。如果转换指针失败，返回 nullptr；如果转换引用失败，抛出 std::bad_cast 异常。有一定的性能开销。

**示例：**
```cpp
class Base {
public:
    virtual ~Base() {}
};
class Derived : public Base {
public:
    void foo() {}
};

Base* b = new Derived();
if (Derived* d = dynamic_cast<Derived*>(b)) {
    d->foo();  // 安全调用
}

Base* b2 = new Base();
Derived* d2 = dynamic_cast<Derived*>(b2);  // 返回 nullptr
```

## 3. const_cast

**应用：** 唯一可以用来移除或添加变量的 const 或 volatile 属性的转换符。

**特点：** 通常用于与遗留且不保证常量正确性的 C 语言 API 交互。修改原本就是常量的变量依然是未定义行为。

**示例：**
```cpp
void func(int* p) {
    *p = 42;
}

const int x = 10;
// func(&x);  // 编译错误
func(const_cast<int*>(&x));  // 移除const属性

// 注意：修改原本是const的变量是未定义行为
```

## 4. reinterpret_cast

**应用：** 处理最底层的位模式重新解释。例如将指针转换为足够大的整数类型，或者在不相关的指针类型之间进行转换。

**特点：** 极其不安全，高度依赖于编译器和底层硬件架构，它不进行任何地址偏移调整。

**示例：**
```cpp
int x = 42;
void* p = &x;
int* q = reinterpret_cast<int*>(p);  // 安全

// 指针转整数
uintptr_t addr = reinterpret_cast<uintptr_t>(p);

// 不相关类型之间的转换（危险）
class A {};
class B {};
A a;
B* b = reinterpret_cast<B*>(&a);  // 危险，可能导致未定义行为
```

## 应用场景总结

| 转换类型 | 主要应用场景 | 安全性 | 性能 |
|---------|------------|--------|------|
| static_cast | 基本类型转换、明确安全的类层次转换 | 中等 | 高 |
| dynamic_cast | 多态类的安全向下转换 | 高 | 中（有运行时开销） |
| const_cast | 与非const API交互 | 低（使用需谨慎） | 高 |
| reinterpret_cast | 底层位模式操作 | 极低（极其危险） | 高 |

## 使用建议

1. **优先使用 static_cast**：对于已知安全的转换，它既高效又安全。
2. **仅在必要时使用 dynamic_cast**：当需要在运行时检查类型时使用，但要注意性能开销。
3. **谨慎使用 const_cast**：只在与遗留代码交互时使用，且不要修改原本是const的变量。
4. **尽量避免 reinterpret_cast**：除非你完全了解底层内存布局和编译器行为。

在 LLVM/MLIR 中，由于禁用了 RTTI，通常使用自定义的 `dyn_cast` 模板（基于 `isa` 和 `classof`）来替代 `dynamic_cast`，以实现类型安全的转换。