# hook-streamlit.py
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.utils.hooks import get_package_paths
import os

def hook(hook_api):
    # 收集 Streamlit 的所有数据
    datas, binaries, hiddenimports = collect_all('streamlit')

    # 额外收集元数据
    try:
        import importlib.metadata as metadata
        dist = metadata.distribution('streamlit')
        # 创建元数据目录结构
        metadata_dir = os.path.join(datas[0][0], 'streamlit-*.dist-info')
        if not os.path.exists(metadata_dir):
            # 手动创建基本的元数据
            pass
    except:
        pass

    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
