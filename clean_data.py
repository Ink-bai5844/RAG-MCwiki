import os
import json

INPUT_DIR = 'output'
OUTPUT_DIR = 'structured_output'

def clean_and_structure(raw_text, title):
    # 1. 基础去噪（保留原有的过滤逻辑）
    lines = raw_text.split('\n')
    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '→' in line or '←' in line:
            continue
        if line.endswith("可以指：") or line.endswith("可以指:"):
            continue
        if "这个消歧义页面列出了" in line or "如果您是通过某条目的内部链接" in line:
            continue
        if line == title and len(valid_lines) == 0:
            continue
        valid_lines.append(line)
        
    # 2. 结构化处理：把文本变成字典
    structured_data = {}
    current_category = "默认分类" # 用于兜底没有分类的条目
    
    for line in valid_lines:
        # 判断当前行是“分类标题”还是“具体条目”
        # 逻辑：如果包含冒号，或者包含数字（针对版本号），或者是句号结尾，说明是具体条目
        if '：' in line or ':' in line or line.endswith('。') or any(char.isdigit() for char in line):
            if current_category not in structured_data:
                structured_data[current_category] = []
            structured_data[current_category].append(line)
        else:
            # 否则，它通常是一个大分类（比如 "Minecraft", "Java版"）
            current_category = line
            if current_category not in structured_data:
                structured_data[current_category] = []
                
    # 过滤掉没有子条目的空分类
    structured_data = {k: v for k, v in structured_data.items() if v}
    
    return structured_data

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.json'):
            continue
            
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_text = data.get("text", "")
            title = data.get("title", "")
            
            # 替换原始的 text 字段为结构化后的 JSON 字典
            data["structured_content"] = clean_and_structure(raw_text, title)
            # 删掉原来那坨难看的文本
            if "text" in data:
                del data["text"]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ 成功结构化: {filename}")
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    main()