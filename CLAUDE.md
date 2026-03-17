# OpenHas — 约束规则

## 类型系统

- 所有领域对象用 `frozen dataclass`（`frozen=True, slots=True`），禁止普通 class。
- 禁止继承（Protocol 做接口约束除外）。
- 可变状态只存在于 `adapters/` 层。
- 使用 `Result[T, E] = tuple[Literal['ok'], T] | tuple[Literal['err'], E]` 做错误处理。

## 项目结构

```
src/
  domain_types/   ← pure dataclass + type alias，零 import 外部库
  core/           ← 纯函数，只 import domain_types/
  adapters/       ← IO 封装（DB、HTTP、文件系统）
  entrypoints/    ← CLI / API route，组装 adapters → core
tests/
  core/           ← 与 core/ 镜像，纯函数测试无需 mock
  adapters/       ← 集成测试，允许 mock
```

- `core/` 禁止 import `adapters/`、禁止 import 任何产生副作用的标准库模块（`os`、`io`、`socket`、`subprocess`）。
- `core/` 中函数签名：输入是 dataclass 或 primitive，输出是 Result。无例外。

## 依赖注入

- 禁止 `dependency-injector` / `fastapi.Depends` 做隐式 DI。依赖通过工厂函数显式传入。
- Protocol 定义只含单个 `__call__`，不做胖接口。

## 错误处理

- `core/` 层禁止 `raise`（仅 `assert` 用于不可能路径）。
- `adapters/` 层在边界处 catch 外部异常，转为 Result 后传入 `core/`。

## Lint + 静态分析

- 每次生成代码后运行 `ruff check --fix && basedpyright`。不通过则重构，禁止 `noqa`。
- McCabe 复杂度上限 5。超过则拆函数，不商量。

## 模块导出

- 所有 `__init__.py` 使用显式 re-export（`as` 语法触发 pyright 的 re-export 检测）。
- 跨模块 import 只通过 `__init__.py`，禁止 `from src.core.auth import _internal_helper`。

## Python 风格

- 禁止 `*args/**kwargs`（除 `adapters/` 中 wrap 外部 API）。显式参数 > 灵活性。
- 字符串拼接用 f-string，禁止 `.format()` 和 `%`。
- 集合操作优先用 comprehension，超过一层嵌套则提取为具名函数。
- 禁止 monkey-patching。测试中用 Protocol 替身，不用 `unittest.mock.patch`。

## 测试

- 使用 Hypothesis 做 property-based test。
- 测试中直接传入 lambda / 闭包 替身，不用 `mock.patch`。
- 每个 test 函数 ≤ 5 行。
