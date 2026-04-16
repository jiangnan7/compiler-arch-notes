# MLIR WalkResult & WalkOrder 详解

## 1. WalkResult

`WalkResult` 表示“这次回调之后 walker 应该怎么办”，主要有两种状态：

- **advance()**
  - 继续遍历，不中断
- **interrupt()**
  - 中断遍历，返回当前结果

### 用法：
- 大部分时候你会 `return WalkResult::advance();`
- 如果在遍历过程中发现了某个条件（比如找到第一个特定 ForOp），可以 `return WalkResult::interrupt();`，整个 walk 会立即停止，外层拿到结果后可以判断

### 坑点 1：
只要回调返回类型是 `WalkResult`，所有路径都必须 return 一个 `WalkResult`。之前的 Illegal instruction 就是因为 lambda 末尾没 return，编译器生成了 unreachable/trap。

## 2. WalkOrder（遍历顺序）

`WalkOrder` 大概就是一个 enum（默认是 PreOrder）：

- **WalkOrder::PreOrder**
  - 先访问父节点 op，再访问它的 region/block 里的子 op
  - 适合做“top-down” 分析，比如你先看 loop 再看里面的 body。

- **WalkOrder::PostOrder**
  - 先访问子 op，再访问父 op
  - 适合做“bottom-up” 分析，比如先分析子表达式，再在父 op 汇总信息（类似 IR 中 classic 后序遍历）。

### 举例：
```cpp
op->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    // 进到父 op 时先执行，再递归到子 op
});

op->walk<WalkOrder::PostOrder>([&](Operation *nested) {
    // 所有子 op 都遍历完后再回到这个 op
});
```

## 3. Operation::walk —— 最常用的一类

典型形式有两种：

### (1) 回调签名：void(Operation *)
```cpp
op->walk([](Operation *nested) {
  // 遍历到每个 nested operation 时都会回调这里
  nested->dump();
});
```

**特点：**
- 回调返回 void，不能中断，一定从头遍历到尾。
- 适合做“只收集信息、不需要提前 stop”的场景。

### (2) 回调签名：WalkResult(Operation *)
```cpp
WalkResult ret = op->walk([&](Operation *nested) -> WalkResult {
  if (isa<dataflow::ForOp>(nested)) {
    // 找到想要的 op，可以中断
    foundForOp = cast<dataflow::ForOp>(nested);
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
});
```

**特点：**
- 允许中断，逻辑更灵活。
- 必须保证所有路径都 return（你刚踩的坑）。

### (3) 指定遍历顺序
两种写法：
```cpp
op->walk<WalkOrder::PreOrder>([](Operation *nested) { ... });
op->walk<WalkOrder::PostOrder>([](Operation *nested) { ... });
```

或者带 WalkResult：
```cpp
op->walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult { ... });
```