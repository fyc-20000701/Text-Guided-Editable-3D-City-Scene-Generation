import json
import requests
import time
import threading
from typing import Dict, Any, Tuple, List
import subprocess
import sys
import os
from pathlib import Path

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "sk-c47d110ce8b347708f2ec4eb3ce86930"

SYSTEM_PROMPT = """你是一个专业的空间建模助手，需要精确解析自然语言描述并生成结构化布局数据。请严格按照以下规则处理：

1. 实体识别：
- 只识别三类实体：water(水域)、building(建筑)、green(绿地)
- 每个独立实体必须分配唯一编号，最小语义单元只能包含一个实体
- 被包围的实体要拆分为8个方位（正东、东南等）的独立实体

2. 布局规范：
- 位置描述使用精确方位词（north、northwest等），绝对不要使用around/surround等模糊表述
- 关系字段指向父节点编号，形成树状结构（根节点为0）
- 形状字段从[rec, ellipse]中选择，默认rec

3. 模型特征：
- 编号与布局文件对应
- 描述字段保持简洁，使用名词短语
- models.json中0号固定为地形预设（如hills/plains），而layout.json中不包含地形预设且从1开始
"""


def generate_prompt(user_input: str) -> str:
    return f"""请严格按以下步骤处理文本，生成两个独立的JSON数组：
<<输入文本>>
{user_input}

<<处理步骤>>
1. 识别所有实体及其类型，被包围的实体拆分为8个方位
2. 建立实体关系树（根节点为0），使用BFS遍历编号
3. 为每个实体生成：
   - 位置（精确方位词）
   - 形状（默认rec）
   - 关系（父节点编号）
4. 生成地形预设（编号0）和模型描述

<<输出格式>>
请先输出layout.json数组，然后输出models.json数组，输出必须严格遵循以下格式：

<<layout.json示例>>
[
  {{{{"num": 1, "entity": "building", "relation": 0, "position": "center", "shape": "rec"}}}},
  // 更多条目...
]

<<models.json示例>>
[
  {{{{"num": 0, "description": "plains"}}}},
  {{{{"num": 1, "description": "high-rise building"}}}},
  // 更多条目...
]

特别注意：
1. 两个JSON数组直接输出，不要嵌套在父对象中
2. 使用标准的JSON格式，不要包含注释
3. 确保所有键使用双引号
4. 数组之间用空行分隔"""

# 新增计时器相关代码
class ApiCallTimer:
    def __init__(self):
        self.start_time = None
        self._event = threading.Event()
        self._thread = None

    def _print_elapsed_time(self):
        while not self._event.is_set():
            elapsed = time.time() - self.start_time
            print(f"\rAPI调用已进行：{elapsed:.1f}秒", end="", flush=True)
            time.sleep(0.1)  # 更细粒度的更新间隔

    def start(self):
        self.start_time = time.time()
        self._event.clear()
        self._thread = threading.Thread(target=self._print_elapsed_time)
        self._thread.start()

    def stop(self):
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        elapsed = time.time() - self.start_time
        print(f"\rAPI调用耗时：{elapsed:.1f}秒        ")  # 清除行尾动态显示

def clean_json_response(text: str) -> str:
    """清理API响应中的常见格式问题"""
    return (
        text.strip()
        .replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
        .replace("//", "")
        .replace("...", "")
    )


def parse_api_response(response_text: str) -> Tuple[List, List]:
    """解析API响应并分离两个JSON数组"""
    cleaned_text = clean_json_response(response_text)
    parts = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]

    if len(parts) < 2:
        raise ValueError(f"响应格式错误，需要两个JSON数组，实际得到{len(parts)}个部分")

    try:
        layout = json.loads(parts[0])
        models = json.loads(parts[1])
        return layout, models
    except json.JSONDecodeError as e:
        print("解析失败的响应内容：")
        print(cleaned_text[:1000])
        raise ValueError("JSON解析失败，请检查响应格式") from e

def call_deepseek_api(prompt: str) -> tuple[list, list]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Cache-Control": "no-cache"
    }
    payload = {
        # DeepSeek-R1
        "model": "deepseek-reasoner",
        # DeepSeek-V3
        #"model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "stream": False
    }
    timer = ApiCallTimer()
    try:
        timer.start()
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        full_response = response.json()

        # 调试输出
        print("完整API响应：")
        print(json.dumps(full_response, indent=2, ensure_ascii=False))

        response_text = full_response["choices"][0]["message"]["content"]
        return parse_api_response(response_text)
    except requests.exceptions.HTTPError as err:
        error_info = f"""HTTP错误详情：
        状态码：{err.response.status_code}
        响应头：{err.response.headers}
        响应体：{err.response.text[:500]}"""
        print(error_info)
        raise
    finally:
        timer.stop()  # 确保任何情况都停止计时器


def validate_data_structure(layout: List, models: List) -> None:
    """验证数据结构完整性"""
    required_layout_keys = {"num", "entity", "relation", "position", "shape"}
    required_model_keys = {"num", "description"}

    for idx, item in enumerate(layout):
        if not isinstance(item, dict):
            raise TypeError(f"layout第{idx}项应为字典类型")
        missing = required_layout_keys - item.keys()
        if missing:
            raise ValueError(f"layout第{idx}项缺少字段：{missing}")

    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            raise TypeError(f"models第{idx}项应为字典类型")
        missing = required_model_keys - item.keys()
        if missing:
            raise ValueError(f"models第{idx}项缺少字段：{missing}")

def save_json(data: List, filename: str) -> None:
    """保存JSON文件并验证格式"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"{filename} 保存成功")
    except Exception as e:
        print(f"保存{filename}失败：{str(e)}")
        raise


def process_description(description: str) -> None:
    prompt = generate_prompt(description)
    try:
        layout_data, models_data = call_deepseek_api(prompt)
        validate_data_structure(layout_data, models_data)
        save_json(layout_data, "layout.json")
        save_json(models_data, "models.json")
    except Exception as e:
        print(f"处理失败：{str(e)}")
        raise


def run_external_scripts():
    """执行外部Python脚本"""
    scripts = ["layout_gen.py"]

    for script in scripts:
        try:
            current_dir = Path(__file__).parent
            script_path = current_dir / script

            if not script_path.exists():
                raise FileNotFoundError(f"脚本不存在：{script_path}")

            print(f"\n正在执行 {script}...")

            # 准备环境变量
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # 启动进程
            with subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
            ) as proc:

                # 实时输出处理
                while True:
                    output = proc.stdout.readline()
                    if output == '' and proc.poll() is not None:
                        break
                    if output:
                        print(f"[{script} OUTPUT] {output.strip()}")

                # 获取剩余输出
                stdout, stderr = proc.communicate()

                # 检查返回码
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode,
                        proc.args,
                        output=stdout,
                        stderr=stderr
                    )

                print(f"执行成功：{script}")

        except Exception as e:
            print(f"执行失败：{script}")
            if hasattr(e, "stderr") and e.stderr:
                print(f"错误信息：{e.stderr}")
            raise

if __name__ == "__main__":
    #user_input = """In a large metropolitan area, there is a central lake surrounded by a ring of commercial buildings. Beyond the commercial area lies a dense forest to the north and a residential zone to the south. To the east of the lake, there is a recreational park with a stadium in the center. Roads connect all the areas seamlessly."""
    try:
        with open('dspt.txt', 'r', encoding='utf-8') as file:
            user_input = file.read()
        process_description(user_input)
        # 后续脚本
        run_external_scripts()
    except FileNotFoundError:
        print("错误：未找到输入文件dspt.txt")
    except Exception as e:
        print(f"程序运行异常：{str(e)}")
    input("Press Enter to continue...")