@echo off
chcp 65001 >nul
echo ============================================================
echo 玻尔兹曼模拟器打包脚本（多文件模式 - 推荐）
echo ============================================================
echo.

echo 正在清理旧文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist BoltzmannSimulator.spec del BoltzmannSimulator.spec

echo 正在打包...
echo.

"E:\Anaconda\Scripts\pyinstaller.exe" --name="BoltzmannSimulator" ^
            --windowed ^
            --noconsole ^
            --add-data="app.py;." ^
            --additional-hooks-dir=. ^
            --collect-all=streamlit ^
            --collect-all=plotly ^
            --collect-all=numpy ^
            --collect-all=scipy ^
            --collect-all=pandas ^
            --collect-all=numba ^
            --collect-all=scipy.constants ^
            --hidden-import=streamlit.web.bootstrap ^
            --clean ^
            --noconfirm ^
            run.py

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo ✅ 打包成功！
    echo ============================================================
    echo.
    echo 可执行文件位置: dist\BoltzmannSimulator\BoltzmannSimulator.exe
    echo.
    echo 📌 重要提示：
    echo    1. 这是多文件模式，整个 dist\BoltzmannSimulator 文件夹需一起移动
    echo    2. 双击 BoltzmannSimulator.exe 运行
    echo    3. 应用会自动打开浏览器（如果失败，请手动访问 http://localhost:8501）
    echo.
    echo ============================================================
) else (
    echo.
    echo ❌ 打包失败，请查看上方错误信息
    echo.
    echo 常见解决方法：
    echo 1. 确保所有包已安装: pip install streamlit plotly numpy scipy pandas numba pyinstaller
    echo 2. 尝试更新 PyInstaller: pip install --upgrade pyinstaller
    echo 3. 检查杀毒软件是否阻止了打包
)

pause
