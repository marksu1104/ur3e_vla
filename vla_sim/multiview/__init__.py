"""MultiView extension of the shared base scene.

``MultiviewSceneCfg`` inherits ``BaseSceneCfg`` and adds fixed top, left, and
right recording cameras. The extension retains its own collection buffer and
H5 additions so canonical collection remains unaffected.
"""
