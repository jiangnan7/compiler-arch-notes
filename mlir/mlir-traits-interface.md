# MLIR 的 Traits 和 Interface 详解

## Traits：描述静态性质

Traits 是在 Operation 定义阶段就绑定的元信息（compile-time property），用于声明 operation 的静态行为和结构约束。

### 特性
- **编译期静态**：traits 一旦在 ODS（Operation Definition Specification）或 C++ 中绑定，op 的行为就固定了。
- **不可在运行时改变**：traits 是类模板的一部分，写死在 Operation 类的元信息里。
- **常用于结构约束**：
  - 操作数/结果数量固定或可变
  - 是否有 side effect
  - 是否 commutative / associative
  - 是否 shape-preserving

### 示例
```tablegen
def MyAddOp : MyDialect_Op<"add",
    [Commutative, NoSideEffect]> {
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}
```

- **这里用到两个 traits**：
  - `Commutative`：说明 `MyAddOp(lhs, rhs) == MyAddOp(rhs, lhs)`
  - `NoSideEffect`：优化器可以安全地删除没有使用的 add op

### C++ 中的 traits 查询
```cpp
if (op.hasTrait<mlir::OpTrait::Commutative>()) {
    // 可以安全交换操作数
}
```

## Interfaces：定义动态行为

Interfaces 是一种多态协议，用于为不同的 op、type、attribute 提供可动态扩展的统一行为。

### 特性
- **运行期可查询**：是否支持某个 interface 以及如何实现是动态的。
- **可为多个 op 提供同一套 API**：
  - 例如，不同算子都可以实现 "InferType" 接口，但具体推导逻辑不同。
- **允许后期扩展**：可以在已有 op 上新增实现，而不用修改 op 本身。

### 示例：InferTypeOpInterface
```tablegen
def InferTypeOpInterface : OpInterface<"InferType"> {
  let methods = [
    InterfaceMethod<
      /*desc=*/"推导输出类型",
      /*retType=*/"LogicalResult",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins MLIRContext:$context,
                    Optional<Location>:$loc,
                    ArrayRef<Type>:$operandsTypes,
                    DictionaryAttr:$attrs,
                    RegionRange:$regions,
                    SmallVectorImpl<Type>&:$inferredReturnTypes)>
  ];
}
```

如果某个 op 实现了该 interface，则只需要提供具体逻辑：

```cpp
LogicalResult MyOp::inferReturnTypes(MLIRContext *ctx,
                                     Optional<Location> loc,
                                     ArrayRef<Type> operandTypes,
                                     DictionaryAttr attrs,
                                     RegionRange regions,
                                     SmallVectorImpl<Type> &inferredReturnTypes) {
    // 根据 operand 类型计算返回类型
    inferredReturnTypes.push_back(operandTypes[0]);
    return success();
}
```

在其他地方就能统一调用：

```cpp
if (auto infer = dyn_cast<InferTypeOpInterface>(op)) {
    SmallVector<Type> inferred;
    if (succeeded(infer.inferReturnTypes(ctx, ...))) {
        ...
    }
}
```

## 核心区别对比

| 特性 | Traits | Interfaces |
|------|--------|------------|
| 定义阶段 | Operation 定义时静态绑定 | Operation 实现时动态选择 |
| 作用 | 声明操作的静态性质和结构约束 | 提供动态行为的统一调用接口 |
| 实现方式 | C++ 模板 + ODS attribute | 虚函数表 + TableGen interface |
| 扩展性 | 需要修改 op 定义 | 新增 interface 或后续实现即可 |
| 查询方式 | `op.hasTrait<>()` | `dyn_cast<SomeInterface>(op)` |
| 典型用途 | 声明 commutative、no side effect、operand count 等元信息 | 类型推导、shape 推导、bufferize、folding 等动态逻辑 |

## 总结

- **Traits = 静态元信息**：优化器根据它做通用的静态优化。
- **Interfaces = 动态多态协议**：让不同 op 共享统一 API，具体实现可差异化。

如果你在 MLIR 做 GPU 后端动态 shape 支持，建议：
- 用 Traits 声明 shape-preserving、commutative 等算子特性
- 用 Interfaces 定义 GPU kernel 的 launch 参数推导、bufferize、shape 计算等逻辑