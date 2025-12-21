# run.py - 用于PyInstaller的启动器（直接 bootstrap 方式）
import sys
import os
import streamlit.web.bootstrap as boot


def main():
    # 让 Streamlit 解压后能定位到 app.py
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_path, 'app.py')

    # 固定端口 & 自动打开浏览器
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"

    boot.load_config_options({
        "browser.serverAddress": "localhost",
        "browser.gatherUsageStats": False
    })

    # 启动 Streamlit 服务（阻塞，直到用户关闭窗口）
    boot.run(app_path, command_line="", args=[])


if __name__ == '__main__':
    main()