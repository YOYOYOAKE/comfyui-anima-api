"""upscale/workflows.py 单测：超分图 builder（sharp 链，纯图像处理无扩散节点）。"""

from comfyui_anima_api.upscale import build_upscale_workflow


class TestUpscaleWorkflow:
    def test_sharp_chain_only(self) -> None:
        # 5 节点线性链：底图 -> us_model/us/us_half -> 落盘；无 KSampler，纯图像处理
        wf = build_upscale_workflow("u [temp]")
        assert set(wf) == {"img", "us_model", "us", "us_half", "save"}
        assert wf["img"]["inputs"]["image"] == "u [temp]"
        assert wf["us"]["inputs"]["image"] == ["img", 0]
        assert wf["us_half"]["inputs"]["scale_by"] == 0.5
        assert wf["save"]["inputs"]["images"] == ["us_half", 0]
        assert wf["save"]["inputs"]["filename_prefix"] == "upscale"

    def test_filename_prefix_threaded_through(self) -> None:
        wf = build_upscale_workflow("u [temp]", "upscale/t1/001")
        assert wf["save"]["inputs"]["filename_prefix"] == "upscale/t1/001"
