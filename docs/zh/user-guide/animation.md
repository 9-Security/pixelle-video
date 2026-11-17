# 动画生成

**动画生成** 流水线可以把一句故事创意（或一篇现成的故事）变成带旁白配音的动画短片。

## 工作原理

```
故事创意 ──► LLM：生成结构化动画剧本
               ├─ 角色设定（每个角色固定的英文外观描述）
               └─ 分镜：旁白 / 画面 / 出场角色 / 动作 / 运镜
          ──► 每个分镜提示词 = 画风预设 + 分镜画面 + 角色外观
          ──► 逐个分镜：
                TTS 旁白配音
                camera 模式  ：文生图 ─► 推拉/平移运镜动画（Ken Burns）
                ai_video 模式：文生视频（提示词附带动作与运镜描述）
                字幕模板叠加
          ──► 拼接 + 可选背景音乐 ──► final.mp4 + animation_script.json
```

角色一致性来自「每个分镜都重复完整角色描述 + 固定画风前缀」，不依赖 LoRA 或参考图，可用于任意图像模型。

## 参数说明

| 参数 | 可选值 | 说明 |
|---|---|---|
| `mode` | `generate` / `adapt` | `generate`：根据创意编写新故事；`adapt`：把已有故事拆成分镜并保留原文 |
| `motion_mode` | `camera` / `ai_video` | `camera` 支持任意图片工作流，成本最低；`ai_video` 需要视频工作流与 `video_*` 模板 |
| `art_style` | `anime`、`3d_cartoon`、`watercolor`、`chinese_ink`、`pixel_art`、`clay`、`flat_vector`、`comic` | 填写 `custom_style` 时以自定义为准 |
| `camera_motion` | `auto`、`zoom_in`、`zoom_out`、`pan_left`、`pan_right`、`pan_up`、`pan_down`、`static` | `auto` 使用 LLM 为每个分镜选择的运镜 |
| `camera_intensity` | 0.0 – 0.5 | 默认 0.15；`image_*` 模板会自动减半，因为字幕会随画面移动 |

推荐使用 `video_*` 模板：画面在下层运动，字幕保持固定。

## Web 界面

打开 **🎞️ 动画生成** 标签页，输入故事创意，选择画风、动画方式、图片（或视频）工作流与配音音色，点击 **生成动画**。生成完成后，视频下方会显示动画剧本（角色与分镜），并可下载 JSON。

## API

```bash
curl -X POST http://localhost:8000/api/animation/generate/async \
  -H "Content-Type: application/json" \
  -d '{
        "text": "一只怕黑的小狐狸，为了帮森林找回月亮而踏上冒险",
        "n_scenes": 5,
        "art_style": "watercolor",
        "motion_mode": "camera",
        "media_workflow": "runninghub/image_flux.json",
        "frame_template": "1080x1920/video_default.html",
        "tts_voice": "zh-CN-XiaoxiaoNeural"
      }'
```

轮询 `GET /api/tasks/{task_id}`，结果包含 `video_url` 与 `script_url`。`GET /api/animation/styles` 可列出画风预设。

## 输出

`output/<task_id>/` 下包含 `final.mp4`、`frames/`、`storyboard.json`、`metadata.json` 与 `animation_script.json`（剧本、画风提示词、运镜与最终提示词），便于复现。
