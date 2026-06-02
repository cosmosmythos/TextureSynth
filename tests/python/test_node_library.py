"""
NodeLibrary surface: every node must have id + outputs, enum is exact.
"""
import texturesynth_core as tc


EXPECTED_FORMATS = {"Mono", "UV", "RGB", "RGBA", "ID", "Metadata"}


def test_library_nonempty_after_init(engine):
    lib = engine.node_library()
    assert len(lib.all()) > 0, "node library is empty after init"


def test_each_node_has_id_and_outputs(engine):
    lib = engine.node_library()
    for type_id, node_type in lib.all().items():
        assert isinstance(type_id, str) and type_id, f"bad type_id: {type_id!r}"
        assert node_type.id == type_id, f"id mismatch: {type_id} vs {node_type.id}"
        assert isinstance(node_type.outputs, list), f"outputs not a list for {type_id}"
        for sock in node_type.outputs:
            assert sock.name, f"output socket missing name in {type_id}"
            assert sock.type in (
                tc.SocketType.Float,
                tc.SocketType.Vec4,
                tc.SocketType.Sampler2D,
            ), f"unknown socket type on {type_id}.{sock.name}: {sock.type}"


def test_channel_format_enum_has_six_values():
    """User constraint: enum must stay at Mono/UV/RGB/RGBA/ID/Metadata."""
    members = {m for m in dir(tc.ChannelFormat) if not m.startswith("_")}
    assert members == EXPECTED_FORMATS, f"ChannelFormat drifted: {members}"


def test_socket_type_enum_complete():
    members = {m for m in dir(tc.SocketType) if not m.startswith("_")}
    assert members == {"Float", "Vec4", "Sampler2D"}, f"SocketType drifted: {members}"


def test_error_code_enum_complete():
    """The structured error enum has the expected set of codes."""
    members = {m for m in dir(tc.EngineErrorCode) if not m.startswith("_")}
    expected = {
        "None", "InitFailed", "ShutdownFailed",
        "GraphValidation", "GraphCompile", "ShaderCompile", "PipelineCreation",
        "NoPipeline", "StaleGeneration", "SubmitRingFull",
        "ParamUnknownNode", "ParamUnknownName",
        "ImageUploadShape", "ImageUploadRingFull", "ImageUploadOOM", "ImageReleaseUnknown",
        "VulkanCommand", "Unknown",
    }
    assert members == expected, f"EngineErrorCode drifted: missing={expected - members}, extra={members - expected}"


def test_phase_enum_complete():
    members = {m for m in dir(tc.EnginePhase) if not m.startswith("_")}
    expected = {
        "Idle", "Init", "GraphSubmit", "GraphCompileFinish", "ParamUpdate",
        "ImageUpload", "ImageRelease", "Submit", "Readback", "Shutdown",
    }
    assert members == expected, f"EnginePhase drifted: missing={expected - members}, extra={members - expected}"
