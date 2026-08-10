"""Multiview extension package.

Adds two extra third-person cameras (left/right) to the canonical
vla_sim scene without modifying any file under vla_sim/.

Assumed layout (multiview/ sits next to vla_sim/ and scripts/ at the
project root):

    project_root/
        vla_sim/
            config.py
            scene.py
            runtime.py
            data_collector.py
            ...
        scripts/
            collect_demos.py
        multiview/
            __init__.py
            config_multiview.py
            scene_multiview.py
            runtime_multiview.py
            data_collector_multiview.py
            collect_demos_multiview.py
"""