# C++ 智能指针详解

C++ 中共有三种主要的智能指针，都定义在 `<memory>` 头文件中，用于自动管理动态内存，防止内存泄漏和悬空指针问题。

## 🟢 1. std::unique_ptr

**独占所有权智能指针**

### 特性：
- 一个对象只能被一个 unique_ptr 拥有；
- 不允许拷贝（copy），但允许移动（move）；
- 当 unique_ptr 被销毁时，会自动释放所管理的对象；
- 适合表达“唯一拥有权”的语义。

### 常用操作：
```cpp
#include <memory>

std::unique_ptr<int> p1 = std::make_unique<int>(10);
std::unique_ptr<int> p2 = std::move(p1);  // 转移所有权
// p1 现在为空
```

### 适用场景：
- 独占资源（文件句柄、网络连接、锁等）；
- 避免意外的拷贝。

## 🟣 2. std::shared_ptr

**共享所有权智能指针**

### 特性：
- 多个 shared_ptr 可以共同拥有同一个对象；
- 内部通过引用计数（reference count）实现；
- 当最后一个 shared_ptr 被销毁时，资源自动释放；
- 线程安全地修改引用计数，但对象本身不是线程安全的。

### 常用操作：
```cpp
#include <memory>

std::shared_ptr<int> p1 = std::make_shared<int>(20);
std::shared_ptr<int> p2 = p1;  // 引用计数 +1
std::cout << p1.use_count();   // 输出 2
```

### 适用场景：
- 需要多个对象共享同一资源；
- 生命周期不容易明确时。

### 注意：
- **循环引用（cyclic reference）** 会导致内存泄漏。
  （即两个对象的 shared_ptr 互相指向）

## 🟡 3. std::weak_ptr

**弱引用智能指针**

### 特性：
- 不能直接拥有对象；
- 依附于 shared_ptr，不会增加引用计数；
- 可以用 lock() 临时获取一个 shared_ptr；
- 用于打破循环引用。

### 常用操作：
```cpp
#include <memory>

std::shared_ptr<int> sp = std::make_shared<int>(30);
std::weak_ptr<int> wp = sp;

if (auto temp = wp.lock()) {
    std::cout << *temp << std::endl;  // 安全访问
}
```

### 适用场景：
- 解决循环引用；
- 缓存、观察者模式等弱依赖关系。

## 🔵 对比总结表：

| 特性 | unique_ptr | shared_ptr | weak_ptr |
|------|------------|------------|----------|
| 拥有权 | 独占 | 共享 | 弱引用（无拥有权） |
| 引用计数 | 否 | 是 | 依附于 shared_ptr |
| 可拷贝 | 否（只能移动） | 是 | 可拷贝 |
| 线程安全 | 否（对象层面） | 引用计数安全 | 否 |
| 常见用途 | 资源独占、RAII | 多方共享 | 避免循环引用 |