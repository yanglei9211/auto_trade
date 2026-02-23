#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT API 调用脚本
基于内部 AI Gateway 实现文本对话功能
支持文本对话和图片理解
"""

import requests
import json
import base64
import os
from pathlib import Path


def get_available_models():
    """
    获取可用的模型列表
    
    Returns:
        可用模型列表
    """
    api_url = "https://ai-gateway-internal.dp.tech/v1/models"
    api_key = "sk-sjqhZZP8H9LNaIc4Utxikg"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print("\n[INFO] 可用的模型列表:")
        print("=" * 60)
        if "data" in result:
            for model in result["data"]:
                model_id = model.get("id", "unknown")
                print(f"  - {model_id}")
            return [model.get("id") for model in result["data"]]
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return []
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}")
        return []


def encode_image_to_base64(image_path):
    """
    将本地图片文件编码为 base64 字符串
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_mime_type(image_path):
    """
    根据文件扩展名获取 MIME 类型
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        MIME 类型字符串
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')


def call_gpt_api(user_message, model_name=None, image_path=None):
    """
    调用 GPT API 获取响应
    
    Args:
        user_message: 用户输入的消息文本
        model_name: 模型名称，如果为 None 则使用默认模型
        image_path: 图片文件路径（可选），如果提供则会发送图片进行理解
        
    Returns:
        AI 的回复内容
    """
    # API 配置
    api_url = "https://ai-gateway-internal.dp.tech/v1/chat/completions"
    api_key = "sk-sjqhZZP8H9LNaIc4Utxikg"
    
    # 如果没有指定模型，使用默认值
    if model_name is None:
        # 常见的 OpenAI 兼容模型名称
        model = "gpt-5.2"  # 可能的模型名称，实际需要根据可用列表调整
    else:
        model = model_name
    
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 构建消息内容
    if image_path and os.path.exists(image_path):
        # 如果提供了图片，构建包含图片的消息
        print(f"[INFO] 正在编码图片: {image_path}")
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        message_content = [
            {
                "type": "text",
                "text": user_message
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]
    else:
        # 纯文本消息
        message_content = user_message
    
    # 构建请求体
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ]
    }
    
    try:
        # 发送 POST 请求
        print(f"[DEBUG] 请求 URL: {api_url}")
        print(f"[DEBUG] 请求头: {headers}")
        print(f"[DEBUG] 请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        # 打印响应状态和内容
        print(f"[DEBUG] 响应状态码: {response.status_code}")
        print(f"[DEBUG] 响应内容: {response.text[:500]}")
        
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        # 提取 AI 回复内容
        if "choices" in result and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]
            return ai_response
        else:
            return f"错误: 未能获取有效响应\n完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}"
            
    except requests.exceptions.HTTPError as e:
        # HTTP 错误，尝试获取详细错误信息
        try:
            error_detail = response.json()
            return f"HTTP 错误 {response.status_code}: {json.dumps(error_detail, ensure_ascii=False, indent=2)}"
        except:
            return f"HTTP 错误 {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"请求失败: {str(e)}"
    except json.JSONDecodeError as e:
        return f"响应解析失败: {str(e)}\n响应内容: {response.text}"
    except Exception as e:
        return f"未知错误: {str(e)}"


def main():
    """主函数"""
    
    # ========== 配置区域 ==========
    
    # 模式选择: 'text' 或 'image'
    mode = 'text'  # 改为 'image' 启用图片理解模式
    
    # 文本模式配置
    text_content = """
    你好，请自我介绍一下，你是什么模型，来自哪家公司，在哪个国家

    """
    
    # 图片模式配置
    image_file_path = "/Users/dp/Downloads/prtsc_2026-02-11-1.png"
    image_prompt = "请详细描述这张图片的内容，包括图片中的主要元素、场景、文字等信息。"
    
    # 模型配置
    model_name = 'gpt-5.2'
    
    # ========== 执行区域 ==========
    
    print("=" * 60)
    print("GPT API 调用测试")
    print("=" * 60)
    
    # 首先获取可用模型列表
    available_models = get_available_models()
    
    if mode == 'text':
        # 文本模式
        print(f"\n[模式] 文本对话")
        print(f"[模型] {model_name}")
        print(f"\n发送的文本内容:\n{text_content}")
        print("\n正在调用 AI 进行回复...\n")
        
        response = call_gpt_api(text_content, model_name)
        
    elif mode == 'image':
        # 图片模式
        print(f"\n[模式] 图片理解")
        print(f"[模型] {model_name}")
        print(f"[图片] {image_file_path}")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_file_path):
            print(f"\n[错误] 图片文件不存在: {image_file_path}")
            print("请修改 image_file_path 变量为实际的图片路径")
            return
        
        print(f"[提示] {image_prompt}")
        print("\n正在调用 AI 进行图片理解...\n")
        
        response = call_gpt_api(image_prompt, model_name, image_file_path)
    
    else:
        print(f"\n[错误] 未知模式: {mode}")
        return
    
    print("=" * 60)
    print("AI 的回复:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
