# C++ 多态详解

## 🧩 一、什么是多态（Polymorphism）

多态的字面意思是 "多种形态"。
在 C++ 中，它表示 相同的接口、不同的实现行为。
也就是说——
"父类指针（或引用）指向子类对象时，通过虚函数调用不同的实现。"

## 🧱 二、多态的三种类型（实现方式）

### 1️⃣ 编译时多态（静态多态）

在编译阶段就能确定调用哪个函数。

**实现方式：**
- 函数重载（Function Overloading）
- 运算符重载（Operator Overloading）
- 模板（Template）

**📘 例子：**
```cpp
#include <iostream>
using namespace std;

void print(int x) { cout << "int: " << x << endl; }
void print(double x) { cout << "double: " << x << endl; }

int main() {
    print(5);     // 调用 print(int)
    print(3.14);  // 调用 print(double)
}
```

**👉 在编译时编译器就能确定具体调用哪个函数。**

### 2️⃣ 运行时多态（动态多态）

运行时根据实际对象类型决定调用哪个函数。
这是面向对象的"核心"多态形式。

**实现条件：**
1. 继承（Inheritance）
2. 基类函数声明为 virtual
3. 使用基类指针或引用调用函数

**示例：**
```cpp
#include <iostream>
using namespace std;

class Base {
public:
virtual void show() { cout << "Base class" << endl; }
virtual ~Base() {}  // 析构函数也要虚拟化，防止内存泄漏
};

class Derived : public Base {
public:
void show() override { cout << "Derived class" << endl; }
};

int main() {
    Base* b = new Derived();
    b->show();   // 输出：Derived class（运行时决定）
    delete b;
}
```

**🔍 原理解析：**
- 每个带 virtual 函数的类都会生成一个 虚函数表（vtable）
- 对象中存有指向该表的 vptr
- 调用虚函数时，会通过 vptr 查表找到对应函数的实际地址

### 3️⃣ 接口多态（抽象类）

如果你希望子类必须实现某些行为，就用 纯虚函数（pure virtual function）。

**示例：**
```cpp
class Shape {
public:
virtual void draw() = 0;  // 纯虚函数
virtual ~Shape() {}
};

class Circle : public Shape {
public:
void draw() override { cout << "Drawing Circle" << endl; }
};

class Square : public Shape {
public:
void draw() override { cout << "Drawing Square" << endl; }
};

int main() {
    Shape* s1 = new Circle();
    Shape* s2 = new Square();
    s1->draw();  // 输出：Drawing Circle
    s2->draw();  // 输出：Drawing Square
    delete s1;
    delete s2;
}
```

**✅ Shape 是抽象类，不能实例化，只能作为接口；**
**✅ 子类必须实现所有纯虚函数。**

## ⚙️ 三、运行时多态的关键机制总结

| 条件 | 说明 |
|------|------|
| 有继承关系 | 必须有基类和派生类 |
| 函数为虚函数 | 使用 virtual 声明 |
| 使用基类指针/引用 | 指向子类对象 |
| 通过该指针/引用调用函数 | 动态绑定到子类函数 |

## 🧠 四、补充：静态 vs 动态多态对比

| 项目 | 静态多态 | 动态多态 |
|------|----------|----------|
| 实现方式 | 函数/运算符重载、模板 | 虚函数机制 |
| 绑定时间 | 编译期 | 运行期 |
| 性能 | 无开销 | 有少量虚表查找开销 |
| 扩展性 | 较差 | 较强 |
| 示例 | print(int/double) | Base* b = new Derived() |