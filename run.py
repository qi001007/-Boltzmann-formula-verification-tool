import sys
import os
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

def main():
    # === 修复控制台输出问题 ===
    if getattr(sys, 'frozen', False):
        # 打包后重定向输出到日志文件
        log_file = os.path.join(os.path.dirname(sys.executable), 'app.log')
        sys.stdout = open(log_file, 'w', encoding='utf-8')
        sys.stderr = sys.stdout
    # =========================
    # 设置环境变量
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"

    # 路径处理
    if getattr(sys, 'frozen', False):
        # 打包后模式
        base = sys._MEIPASS
        app_path = os.path.join(base, 'app.py')
    else:
        # 开发模式
        app_path = os.path.abspath('app.py')

    try:
        # 直接导入并运行
        import streamlit.web.bootstrap as boot

        # 使用更简单的调用方式
        boot.run(
            main_script_path=app_path,
            is_hello=False,
            args=[],
            flag_options={
                'server.headless': 'false',
                'server.port': 8501,
                'server.address': '127.0.0.1'
            }
        )
    except Exception as e:
        # 如果仍然失败，提供详细错误信息
        print(f"启动失败: {e}")
        print(f"Python路径: {sys.executable}")
        print(f"应用路径: {app_path}")
        if getattr(sys, 'frozen', False):
            print(f"MEIPASS: {sys._MEIPASS}")
        input("按回车键退出...")

if __name__ == '__main__':
    main()
