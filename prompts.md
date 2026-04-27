
现在我希望把FCPE 1:1复刻到这个仓库里。
包括所有完整的预处理、后处理逻辑和整个深度学习模型的全部内容。
注意：
- 不允许进行任何的简化，这是一个生产上需要使用的，为了抛弃Python运行时的移植；
- 不允许改变任何逻辑，包括数值精度、运算使用的类型等等。同时你应该和打印日志对齐每一步的精度，因为部分mlx-rs的API即使输入相同，输出精度也和Python的API不同
- 输出结果应当和Python的torch实现（FCPE）和mlx实现（fcpe-mlxrs）尽一切可能对齐，包括你应该检查最后一层Latent vector的对齐精度
- 不允许为了实现方便做任何的妥协。

其他库也必须参考库的本地的源码即可
mlx-rs使用
```toml
mlx-rs = { git = "https://github.com/blossom-slopware/mlx-rs.git", rev = "b81194a47c2c6ba4ecb0cf370e4f0d941f52dd5d", features = ["accelerate", "metal"] }
```
使用这个音频进行测试：/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav
checkpoint: /Users/daisy/develop/fcpe-mlxrs/checkpoint/fcpe.pt
不允许进入plan mode

使用任何你认为合适的第三方库。
如果你不清楚某个库的API，你需要查看源码或者文档。不要上网查文档，本地的库缓存有源码

注意.pt权重可能需要格式转换才能被mlx读取。

---

将python和Rust的输出f0曲线画在一张图中，用不同的颜色。注意不发声段不要画出来