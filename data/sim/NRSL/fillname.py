import os
import sys
import argparse

def write_image_names_to_file(image_folder):
    """
    将指定文件夹中的图像文件名（不含后缀）写入到 names.txt 文件中
    每个文件名重复写两次，格式如：name1 name1
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建图像文件夹的完整路径
    image_folder_path = os.path.join(script_dir, image_folder)

    # 检查图像文件夹是否存在
    if not os.path.exists(image_folder_path):
        print(f"错误：文件夹 '{image_folder}' 不存在于当前目录中")
        return False

    if not os.path.isdir(image_folder_path):
        print(f"错误：'{image_folder}' 不是文件夹")
        return False

    # 支持的图像文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 获取文件夹中的所有文件
    all_files = os.listdir(image_folder_path)

    # 筛选出图像文件
    image_files = []
    for file in all_files:
        # 获取文件扩展名
        _, ext = os.path.splitext(file)
        if ext.lower() in image_extensions:
            # 获取不带扩展名的文件名
            file_name_without_ext = os.path.splitext(file)[0]
            image_files.append(file_name_without_ext)

    # 如果没有找到图像文件
    if not image_files:
        print(f"警告：在文件夹 '{image_folder}' 中没有找到图像文件")
        return False

    # 对文件名进行排序（可选，可根据需要调整）
    image_files.sort()

    # 构建输出文件路径
    output_file_path = os.path.join(script_dir, 'names.txt')

    # 写入文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for name in image_files:
                # 每行写入两次文件名，用空格分隔
                f.write(f"{name} {name}\n")

        print(f"成功将 {len(image_files)} 个图像文件名写入到 names.txt")
        return True

    except Exception as e:
        print(f"写入文件时出错：{e}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='将图像文件夹中的文件名写入到 names.txt 文件中'
    )
    parser.add_argument(
        'folder',
        help='当前目录下的图像文件夹名'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    write_image_names_to_file(args.folder)

if __name__ == "__main__":
    # 如果没有提供命令行参数，显示使用说明
    if len(sys.argv) == 1:
        print("使用方法：python fillname.py <图像文件夹名>")
        print("示例：python fillname.py images")
    else:
        main()