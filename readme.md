1. 使用如下JSON进行导入：
```json
{
  "mcpServers": {
    "kntagrag": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/NatsUIJM/DataExMCP.git",
        "kntagrag"
      ],
      "env": {
        "DASHSCOPE_API_KEY": "你的API密钥"
      }
    }
  }
}
```
2. 修改API密钥为正确数据；
3. 修改超时时间为600秒；
4. 修改镜像源；
5. 运行。