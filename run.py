# run.py - 用于PyInstaller的启动器
import sys
import os
import subprocess


def main():
    # 获取临时目录（PyInstaller解压路径）
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    # 设置环境变量
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

    # 启动Streamlit
    app_path = os.path.join(base_path, 'app.py')
    subprocess.run(['streamlit', 'run', app_path, '--server.port=8501'])


if __name__ == '__main__':
    main()
