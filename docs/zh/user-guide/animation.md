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
| `motion_mode` | `parallax` / `camera` / `ai_video` | `camera` 支持任意图片工作流，成本最低；`ai_video` 需要视频工作流与 `video_*` 模板 |
| `art_style` | `anime`、`3d_cartoon`、`watercolor`、`chinese_ink`、`pixel_art`、`clay`、`flat_vector`、`comic` | 填写 `custom_style` 时以自定义为准 |
| `camera_motion` | `auto`、`zoom_in`、`zoom_out`、`pan_left`、`pan_right`、`pan_up`、`pan_down`、`static` | `auto` 使用 LLM 为每个分镜选择的运镜 |
| `camera_intensity` | 0.0 – 0.5 | 默认 0.15；`image_*` 模板会自动减半，因为字幕会随画面移动 |

推荐使用 `video_*` 模板：画面在下层运动，字幕保持固定。
默认的 `video_fullscreen.html`（1080x1920，另有 1920x1080）让动画铺满整个画面，并加上上下渐层暗角保证标题与字幕清晰（可调整 `title_size`、`subtitle_size`、`accent_color`）。

每个角色有 `name`（英文代号，用于图像提示词）与 `display_name`（旁白中使用、与故事同语言的名字）；若旁白中出现英文代号，会自动替换为 `display_name`，确保名称一致。

## 2.5D 视差模式（默认）

`motion_mode="parallax"` 在 CPU 上免费把每张静态图变成「活」的场景：

1. **分层**：用开源的 IS-Net 动漫分割模型（ONNX，约 176 MB，首次自动下载到 `~/.cache/pixelle-video/models`
   或 `$PIXELLE_MODEL_DIR`，并以 SHA-256 校验）把角色抠出；背景的空洞用 OpenCV 修补并轻微模糊。
2. **视差运镜**：背景与角色按同一个运镜、以不同速度移动，产生前后层次。
3. **角色待机动作**：以脚底为支点的轻微呼吸与上下起伏（`character_motion`）。
4. **环境粒子**：`fireflies`、`stars`、`snow`、`rain`、`petals`、`dust`、`embers`、`bubbles`；
   由 LLM 按分镜选择（`ambient_effect="auto"`），或整支视频固定一种。
5. **收尾**：暗角、柔和光线脉动，以及分镜之间的短淡入淡出。

若分割不可用（首次下载时无网络，或遮罩不可用），该分镜会退回单层运镜加粒子，不会让整支视频失败。
角色仍不会走路或说话，这需要搭配视频模型的 `motion_mode="ai_video"`。

## 免费方案：Gemini + Cloudflare Workers AI

1. **LLM**：系统配置 → LLM 预设选 **Gemini**，填入 Google AI Studio API Key
   （Base URL `https://generativelanguage.googleapis.com/v1beta/openai/`，模型 `gemini-flash-latest`）。
2. **图片**：系统配置 → API 媒体模型 → **Cloudflare Workers AI**，填入 Account ID 与具备 *Workers AI* 权限的 API Token；
   然后在动画生成页选择来源 **API 模型** 与 `flux-1-schnell - API Cloudflare`（工作流 key：`api/cloudflare/flux-1-schnell`）。
3. **配音**：本地 Edge-TTS 无需 Key。

| 工作流 key | 模型 | 说明 |
|---|---|---|
| `api/cloudflare/flux-1-schnell` | `@cf/black-forest-labs/flux-1-schnell` | 画质最佳；固定方图输出，自动居中裁切为模板尺寸 |
| `api/cloudflare/sdxl-lightning` | `@cf/bytedance/stable-diffusion-xl-lightning` | 速度快；支持指定宽高 |
| `api/cloudflare/sdxl-base-1.0` | `@cf/stabilityai/stable-diffusion-xl-base-1.0` | 较慢；支持指定宽高 |
| `api/cloudflare/dreamshaper-8-lcm` | `@cf/lykon/dreamshaper-8-lcm` | 支持指定宽高 |

遇到限流（HTTP 429）或 5xx 会指数退避重试；401/403 会直接报错。大批量生成前请先确认 Cloudflare 当前的免费额度。

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
        "motion_mode": "parallax",
        "media_workflow": "api/cloudflare/flux-1-schnell",
        "frame_template": "1080x1920/video_fullscreen.html",
        "tts_voice": "zh-CN-XiaoxiaoNeural"
      }'
```

轮询 `GET /api/tasks/{task_id}`，结果包含 `video_url` 与 `script_url`。`GET /api/animation/styles` 可列出画风预设。

## 输出

`output/<task_id>/` 下包含 `final.mp4`、`frames/`、`storyboard.json`、`metadata.json` 与 `animation_script.json`（剧本、画风提示词、运镜与最终提示词），便于复现。
