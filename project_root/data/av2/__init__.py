"""AV2 support modules that remain independent of the AV2 SDK.

Keep this initializer free of optional dependencies.  The path and GPU
preflight utilities must be importable before a numerical environment exists;
consumers should import concrete APIs from their submodules, for example
``data.av2.trajectory_transform``.
"""
