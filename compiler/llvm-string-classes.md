# LLVM 的字符串处理类与 std::string 的区别

## 常见的 LLVM 字符串处理类

1. `llvm::StringRef`
2. `llvm::SmallString<N>`
3. `llvm::Twine`
4. `llvm::StringMap<T> / StringSet`
5. `llvm::StringLiteral`

## 与 std::string 的区别

主要围绕 "是否 owning / 是否在堆上 / 是否为高性能工具类"：

### 1. StringRef

- **非 owning 的「字符串视图」**：只保存 `const char* + size_t length`
- **不分配内存，不负责释放**
- **适合参数传递、切片、解析等操作**

**优势：**
- 传参零拷贝，比 std::string（可能要拷贝）轻很多
- 可以指向 C 字符串、std::string、缓冲区等任意内存

### 2. SmallString<N>

- **带小缓冲区的 std::string 替代**：
  - 小于等于 N 的字符串放在 栈内联存储（无堆分配）
  - 超过 N 才回退到堆

**优势：**
- 大量短字符串场景（比如 IR 名字），能减少 malloc/free
- 比普通 std::string 内存碎片更少，cache 友好

### 3. Twine

- **懒拼接字符串工具**：
  ```cpp
  errs() << "error in " << file << ":" << line;
  ```
  这堆东西可以被打包成一个 Twine，在最终写入时一次性遍历，中间不产生临时字符串

**优势：**
- 避免多次 + 拼接带来的中间 std::string 临时对象
- 主要用在打印、日志、错误信息上

## 总结

- **std::string**：拥有字符串 + 一般用途
- **LLVM 系列类**：围绕 性能和内存分配模式优化，强调：
  - 尽量不分配（StringRef）
  - 小对象栈内存（SmallString）
  - 避免中间临时（Twine）