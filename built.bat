# 创建正确的批处理文件
@'
@echo off
chcp 65001 >nul
echo 正在打包玻尔兹曼模拟器...
echo 请确保已安装: pip install pyinstaller
echo.

pyinstaller --name="BoltzmannSimulator" ^
            --onefile ^
            --windowed ^
            --add-data="app.py;." ^
            --hidden-import=streamlit ^
            --hidden-import=plotly ^
            --hidden-import=numpy ^
            --hidden-import=scipy ^
            --hidden-import=pandas ^
            --hidden-import=numba ^
            --hidden-import=scipy.constants ^
            --clean ^
            --noconfirm ^
            app.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ 打包成功！
    echo 可执行文件位于: dist\BoltzmannSimulator.exe
    echo.
    echo 请手动运行: dist\BoltzmannSimulator.exe
) else (
    echo.
    echo ❌ 打包失败，请检查错误信息
)

pause
'@ | Out-File -FilePath "build.bat" -Encoding UTF8

echo "build.bat 已创建完成！"
