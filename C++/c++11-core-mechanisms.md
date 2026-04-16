# C++11 核心机制详解

## 一、C++11 的特性

### 核心语法
- **auto 类型推导**：让编译器自动推导变量类型，减少代码冗余
- **Lambda 表达式**：匿名函数，简化函数对象的创建
- **基于范围的 for 循环**：更简洁地遍历容器元素

### 内存管理
- **智能指针**：`std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr` 等，自动管理内存

### 性能优化
- **右值引用（&&）**：用于实现移动语义
- **移动语义（Move Semantics）**：避免不必要的深拷贝
- **完美转发（Perfect Forwarding）**：保持参数的值类别

### 并发编程
- **标准线程库**：`<thread>`, `<mutex>`, `<atomic>` 等

### 其他
- **constexpr**：编译期常量表达式
- **nullptr**：替代 NULL，更安全的空指针表示

## 二、智能指针

智能指针通过 RAII（资源获取即初始化）机制管理堆内存，在对象作用域结束时自动析构并释放内存，避免内存泄漏。主要包括：
- `std::unique_ptr`：独占所有权
- `std::shared_ptr`：共享所有权
- `std::weak_ptr`：弱引用，不增加引用计数

## 三、unique_ptr 如何保证唯一性

`unique_ptr` 通过将**拷贝构造函数（Copy Constructor）和拷贝赋值运算符（Copy Assignment Operator）**显式删除（`= delete`）来实现排他性所有权。要转移所有权，必须显式调用 `std::move()` 触发移动语义。

**示例：**
```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(42);
// std::unique_ptr<int> p2 = p1;  // 编译错误，拷贝构造被删除
std::unique_ptr<int> p2 = std::move(p1);  // 转移所有权
// 此时 p1 为空，p2 拥有对象
```

## 四、shared_ptr 何时析构

`shared_ptr` 基于引用计数管理对象生命周期。内部有一个控制块（Control Block）记录：
- **强引用（Strong ref）**：拥有对象的引用计数
- **弱引用（Weak ref）**：观察对象的引用计数

当最后一个拥有该对象的 `shared_ptr` 被销毁或重置时（即强引用计数降为 0），它会析构指向的对象。当弱引用计数也降为 0 时，控制块本身的内存才会被释放。

**示例：**
```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(42);
{  
    std::shared_ptr<int> p2 = p1;  // 强引用计数变为 2
    std::weak_ptr<int> wp = p1;     // 弱引用计数变为 1
}
// p2 离开作用域，强引用计数变为 1
// 此时 wp 仍然存在，弱引用计数为 1

p1.reset();  // 强引用计数变为 0，对象被析构
// 但控制块仍然存在，因为弱引用计数为 1

// 当 wp 离开作用域，弱引用计数变为 0，控制块被释放
```

## 五、类的成员函数可以当模板吗

**可以**。称为**成员函数模板（Member Function Template）**。允许在普通类或类模板中定义泛型方法。

**限制**：虚函数（Virtual Functions）不能是模板函数，因为编译器在实例化时无法确定 vtable（虚函数表）的大小。

**示例：**
```cpp
class MyClass {
public:
    template<typename T>
    void print(T value) {
        std::cout << value << std::endl;
    }
    
    // 虚函数不能是模板
    // template<typename T>
    // virtual void virtualPrint(T value);  // 编译错误
};
```

## 六、左值右值

### 左值（Lvalue）
- 有持久的内存地址
- 可以被赋值（出现在等号左边）
- 生命周期超出了当前表达式

### 右值（Rvalue）
- 临时的、没有名字的数值或对象（如字面量、函数返回的临时对象）
- 通常将在表达式结束后销毁

### 右值引用的作用
引入右值引用是为了实现移动语义，把即将销毁的临时对象的内存"偷"过来，避免深拷贝开销。

**示例：**
```cpp
// 左值
int x = 42;
int& lref = x;  // 左值引用

// 右值
int&& rref = 100;  // 右值引用
int&& rref2 = std::move(x);  // 将左值转换为右值
```