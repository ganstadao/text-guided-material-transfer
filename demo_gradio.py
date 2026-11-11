"""Optional UI using the same serial engine and run records as the CLI."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4


def main():
    import gradio as gr
    from material_transfer.config import TransferConfig
    from material_transfer.pipeline import MaterialTransfer
    from material_transfer.records import save_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/A.json")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    engine = MaterialTransfer(TransferConfig.load(args.config))

    def generate(target, exemplar, phrase, mask, depth):
        if target is None or exemplar is None:
            raise gr.Error("请提供目标图和材质参考图")
        artifacts, timings = engine.run(target, exemplar, phrase, mask, depth)
        directory = Path("outputs/ui") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
        save_run(directory, artifacts, engine.config, timings, {"material_prompt": phrase, "source": "gradio"})
        return artifacts["result"], str(directory)

    with gr.Blocks() as app:
        gr.Markdown(f"# Material transfer — method {engine.config.method}")
        with gr.Row():
            target = gr.Image(type="pil", label="目标图")
            exemplar = gr.Image(type="pil", label="材质参考图")
        phrase = gr.Textbox(label="参考图中的材质描述（B/C/D 必填）")
        with gr.Row():
            mask = gr.Image(type="pil", image_mode="L", label="目标 mask：白色为编辑区域（推荐上传）")
            depth = gr.Image(type="pil", label="目标深度图（可选）")
        button = gr.Button("生成")
        output = gr.Image(label="结果")
        record = gr.Textbox(label="保存目录")
        button.click(generate, inputs=[target, exemplar, phrase, mask, depth], outputs=[output, record], concurrency_limit=1)
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
