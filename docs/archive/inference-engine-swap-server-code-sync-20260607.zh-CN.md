# 推理端代码同步记录

日期：2026-06-07

## 当前目的

先暂停正式 benchmark 和进一步结构改动，保留旧 OpenAI 兼容推理格式的结果与路径，同时把服务器上此前实际部署/验证过的 RWKV vLLM 推理端代码完整拉回本地，并上传到 GitHub 留下可追溯记录。

## 范围

- 本地评测仓库：`/home/chase/GitHub/rwkv-skills`
- 服务器入口：`chase@47.115.88.183:8222`
- 目标内容：服务器侧此前用于推理端替换验证的 nano-vLLM/RWKV 推理服务代码，而不是 benchmark 输出日志或密钥文件。

## 当前约束

- 不回退当前工作区已有改动。
- 先保留旧 OpenAI 格式路径；vLLM/RWKV contents 批量路径作为高性能推理端继续记录。
- 上传 GitHub 前需要记录远端来源路径、commit/branch、同步方式和本地落点。
