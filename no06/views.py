# >>> no06/views.py 全文 <<<
import os, cv2, json, subprocess
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from django.http  import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .models import Video, DistributionData

# ──────────────────────────────────────────────
# ML / 数値ライブラリ
# ──────────────────────────────────────────────
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster  import KMeans
import plotly.graph_objects as go
import plotly.colors        as pc
from ultralytics import YOLO

pose_model   = YOLO('yolov8n-pose.pt')   # 肩腰 RGB12
detect_model = YOLO('yolov8n.pt')        # BBox 検出

# ──────────────────────────────────────────────
# 主要パス
# ──────────────────────────────────────────────
APP_DIR = settings.BASE_DIR / settings.APP_NAME
VID_DIR = APP_DIR / 'media' / 'videos'
ORI_DIR = APP_DIR / 'media' / 'originals'
DET_DIR = APP_DIR / 'media' / 'detect'
PRO_DIR = APP_DIR / 'media' / 'processed'
for p in (VID_DIR, ORI_DIR, DET_DIR, PRO_DIR):
    p.mkdir(parents=True, exist_ok=True)

# =========================================================
# 基本ビュー
# =========================================================
def index(request):
    """トップページ (Ajax ベース)"""
    return render(request, 'no06/index.html')

# ---------------------------------------------------------
@require_http_methods(['GET'])
def select_videos(request):
    """データ生成用の動画選択フォーム"""
    videos = Video.objects.order_by('-uploaded')
    return render(request, 'no06/select_videos.html', {'videos': videos})

# ---------------------------------------------------------
@csrf_exempt
@require_http_methods(['GET', 'POST'])
def upload_video(request):
    """動画アップロード + 一覧ページ (Ajax アップロード)"""
    if request.method == 'GET':
        vids = Video.objects.order_by('-uploaded')
        return render(request, 'no06/upload_form.html', {'videos': vids})

    vfile = request.FILES.get('video_file')
    if not vfile:
        return JsonResponse({'error': 'Video file is required.'}, status=400)

    save_path = VID_DIR / vfile.name
    with open(save_path, 'wb+') as dst:
        for chunk in vfile.chunks():
            dst.write(chunk)

    Video.objects.create(
        name     = request.POST.get('name') or vfile.name,
        file     = str(save_path.relative_to(settings.BASE_DIR)),
        uploaded = datetime.now(),
    )
    return JsonResponse({'message': f'"{vfile.name}" をアップロードしました。'})

# ---------------------------------------------------------
@csrf_exempt
@require_http_methods(['POST'])
def delete_video(request, vid):
    """
    動画 1 本とその派生ファイルをすべて削除
    - 通常フォーム: トップページへリダイレクト
    - Ajax        : JSON メッセージ
    """
    video = get_object_or_404(Video, id=vid)

    # 元動画ファイル
    (settings.BASE_DIR / video.file).unlink(missing_ok=True)

    # 派生ファイル (サムネ、検出動画など)
    patterns = [
        ORI_DIR.glob(f'{vid}_*.jpg'),
        DET_DIR.glob(f'{vid}_*.mp4'),
        PRO_DIR.glob(f'{vid}_*.mp4'),
    ]
    for itr in patterns:
        for p in itr:
            p.unlink(missing_ok=True)

    # DB レコード削除（CASCADE で DistributionData も消える）
    video.delete()

    # ---- 戻り値を分岐 -----------------------------------
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'message': f'Video {vid} deleted.'})

    # フォーム送信の場合はトップページ (`index`) に戻す
    return redirect('no06:index')

# =========================================================
# 特徴量生成 (YOLO-Pose → RGB12)
# =========================================================
def _get_average_color(img, cx, cy, r=4):
    h, w, _ = img.shape
    x1, x2 = max(0, cx - r), min(w, cx + r)
    y1, y2 = max(0, cy - r), min(h, cy + r)
    patch  = img[y1:y2, x1:x2]
    if patch.size == 0:
        return [0, 0, 0]
    b, g, r = cv2.mean(patch)[:3]
    return [int(r), int(g), int(b)]

@require_http_methods(['GET'])
def generate_data(request):
    """
    /generateData/?video_ids=1,2   (省略時は全動画)
    * 1 fps サンプリング
    * 肩・腰 4 点の RGB → 12-dim
    * DistributionData 保存
    * サムネイル保存
    """
    ids = request.GET.get('video_ids')
    if ids:
        try:
            sel = [int(i) for i in ids.split(',') if i.strip()]
        except ValueError:
            return JsonResponse({'error': 'invalid ids'}, status=400)
        videos = Video.objects.filter(id__in=sel)
    else:
        videos = Video.objects.all()

    inserted = 0
    for video in videos:
        v_path = settings.BASE_DIR / video.file
        if not v_path.exists():
            continue

        cap  = cv2.VideoCapture(str(v_path))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        step = int(fps)
        fn   = 0

        while True:
            ok = cap.grab()
            if not ok:
                break
            if fn % step:
                fn += 1; continue

            _, frame = cap.retrieve()
            ts_ms   = int((fn / fps) * 1000)

            # --- Pose 推定 ---
            res  = pose_model(frame, imgsz=640, verbose=False)[0]
            kpts = getattr(res, 'keypoints', None)
            if kpts is None or kpts.xy is None:
                fn += 1; continue

            xy   = kpts.xy.cpu().numpy()
            conf = kpts.conf.cpu().numpy() if kpts.conf is not None else None
            if xy.shape[0] == 0 or xy.shape[1] < 13:
                fn += 1; continue

            all_pts = []
            for p in range(xy.shape[0]):          # 各人物
                pts = []
                for idx in (5, 6, 11, 12):       # 肩・腰
                    x, y = map(int, xy[p, idx])
                    sc   = conf[p, idx] if conf is not None else 1.0
                    if sc < .3: pts = []; break
                    pts.append((x, y))
                if len(pts) != 4:
                    continue

                rgb12 = []
                for cx, cy in pts:
                    rgb12.extend(_get_average_color(frame, cx, cy))

                DistributionData.objects.update_or_create(
                    video      = video,
                    timestamp  = ts_ms,
                    defaults=dict(
                        tsne_x=0, tsne_y=0, tsne_z=0,
                        feature_vec=json.dumps(rgb12),
                        cluster=-1, thumb_path=''
                    )
                )
                all_pts.extend(pts)
                inserted += 1

            if all_pts:                           # サムネイル (1 枚)
                for cx, cy in all_pts:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                th_name = f'{video.id}_{ts_ms}.jpg'
                th_path = ORI_DIR / th_name
                cv2.imwrite(str(th_path), frame)
                DistributionData.objects.filter(video=video, timestamp=ts_ms)\
                    .update(thumb_path=str(th_path.relative_to(settings.BASE_DIR)))

            fn += 1
        cap.release()

    return JsonResponse({'message': f'データ生成完了 ({inserted} 点)'})

# =========================================================
# t-SNE + KMeans → 3D 分布
# =========================================================
@require_http_methods(['GET'])
def make_distribution(request):
    ids = request.GET.get('video_ids')
    if ids:
        try:
            sel = [int(i) for i in ids.split(',') if i.strip()]
        except ValueError:
            return JsonResponse({'error': 'invalid ids'}, status=400)
        qs = DistributionData.objects.filter(video_id__in=sel).order_by('id')
    else:
        qs = DistributionData.objects.order_by('id')

    if qs.count() < 5:
        return JsonResponse({'error': 'データが足りません'}, status=400)

    X   = np.array([json.loads(d.feature_vec) for d in qs])
    ts  = TSNE(n_components=3, random_state=0,
               perplexity=max(5, min(30, len(X)//2)))
    X3  = ts.fit_transform(X)
    km  = KMeans(n_clusters=5, random_state=0)
    cls = km.fit_predict(X3)

    for d, (x, y, z), c in zip(qs, X3, cls):
        d.tsne_x = float(x); d.tsne_y = float(y); d.tsne_z = float(z)
        d.cluster = int(c)
        d.save(update_fields=['tsne_x', 'tsne_y', 'tsne_z', 'cluster'])

    return JsonResponse({'message': 'クラスタリング完了'})

# ---------------------------------------------------------
@require_http_methods(['GET'])
def show_distribution(request):
    """
    { data:[Scatter3d…], layout:…, videos:[{id,name}…] }
    """
    ids = request.GET.get('video_ids')
    if ids:
        try:
            sel = [int(i) for i in ids.split(',') if i.strip()]
        except ValueError:
            return JsonResponse({'error': 'invalid ids'}, status=400)
        qs = DistributionData.objects.select_related('video')\
                                     .filter(video_id__in=sel)
    else:
        qs = DistributionData.objects.select_related('video')

    if not qs.exists():
        return JsonResponse({'error': 'no data'}, status=400)

    palette = pc.qualitative.Set2
    traces  = []
    for c in range(5):
        pts = [d for d in qs if d.cluster == c]
        if not pts:
            continue
        traces.append(go.Scatter3d(
            mode   = 'markers',
            name   = f'cluster{c}',
            x      = [d.tsne_x for d in pts],
            y      = [d.tsne_y for d in pts],
            z      = [d.tsne_z for d in pts],
            marker = dict(size=3, color=palette[c]),
            customdata=[(d.video.name, d.timestamp, d.thumb_path) for d in pts],
            hovertemplate = '%{customdata[0]}-%{customdata[1]}<extra></extra>',
        ))

    fig = go.Figure(
        data   = traces,
        layout = dict(scene  = dict(xaxis_title='t-SNE-X',
                                    yaxis_title='t-SNE-Y',
                                    zaxis_title='t-SNE-Z'),
                      margin = dict(l=0, r=0, b=0, t=0),
                      hovermode='closest')
    )

    used = {d.video.id: d.video.name for d in qs}
    d    = json.loads(fig.to_json())
    d['videos'] = [{'id': vid, 'name': name} for vid, name in used.items()]
    return JsonResponse(d, safe=False)

# =========================================================
# サムネイル / 画像 / 動画
# =========================================================
@require_http_methods(['GET'])
def get_image(request):
    rel = request.GET.get('path')
    if not rel:
        return JsonResponse({'error': 'Required param "path"'}, status=400)
    absp = settings.BASE_DIR / rel
    if not absp.exists():
        return JsonResponse({'error': 'file not found'}, status=404)
    return FileResponse(open(absp, 'rb'), content_type='image/jpeg')

@require_http_methods(['GET'])
def get_video(request):
    rel = request.GET.get('path')
    if not rel:
        return JsonResponse({'error': 'Required param "path"'}, status=400)
    absp = settings.BASE_DIR / rel
    if not absp.exists():
        return JsonResponse({'error': 'file not found'}, status=404)
    return FileResponse(open(absp, 'rb'), content_type='video/mp4')

@require_http_methods(['GET'])
def get_processed_video(request):
    """
    1 fps のサムネイル連結動画 (肩腰点付き) を返す
    """
    try:
        vid_id = int(request.GET.get('video_id', ''))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    out_path = PRO_DIR / f'{vid_id}_proc.mp4'
    if not out_path.exists():
        frames = sorted(
            ORI_DIR.glob(f'{vid_id}_*.jpg'),
            key=lambda p: int(p.stem.split('_')[1])
        )
        if not frames:
            return JsonResponse({'error': 'frames not found'}, status=404)

        sample = cv2.imread(str(frames[0]))
        h, w, _ = sample.shape
        vw = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            1.0,
            (w, h)
        )
        for f in frames:
            img = cv2.imread(str(f))
            vw.write(img)
        vw.release()

    return FileResponse(open(out_path, 'rb'),
                        as_attachment=False,
                        filename=out_path.name,
                        content_type='video/mp4')

@require_http_methods(['GET'])
def get_thumb_list(request):
    vid = request.GET.get('video_id')
    try:
        vid = int(vid)
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    frames = sorted(
        ORI_DIR.glob(f'{vid}_*.jpg'),
        key=lambda p: int(p.stem.split('_')[1])
    )
    if not frames:
        return JsonResponse({'error': 'frames not found'}, status=404)

    rel = [str(p.relative_to(settings.BASE_DIR)) for p in frames]
    return JsonResponse({'thumbs': rel})

# =========================================================
# ファイル一覧 (デバッグ用)
# =========================================================
def show_directory(request):
    ext_ok = {'.py', '.html', '.js', '.css'}
    files = []
    for root, dirs, fns in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('__')]
        for f in fns:
            if Path(f).suffix in ext_ok:
                files.append(str(Path(root).relative_to(APP_DIR) / f))
    files.sort()
    lis = ''.join(f'<li>{Path(p)}</li>' for p in files)
    return HttpResponse(f'<h2>no06 内のファイル ({len(files)})</h2>'
                        f'<ul style="line-height:1.4em">{lis}</ul>')

# =========================================================
# BBox 検出動画
# =========================================================
@require_http_methods(['GET'])
def select_detect(request):
    vids = Video.objects.order_by('-uploaded')
    return render(request, 'no06/select_detect.html', {'videos': vids})

@require_http_methods(['GET'])
def bbox_detect(request):
    """
    /bboxDetect/?video_id=xx
    OpenCV で一時 mp4 → ffmpeg H.264 + faststart
    """
    try:
        vid_id = int(request.GET.get('video_id', ''))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    try:
        video = Video.objects.get(id=vid_id)
    except Video.DoesNotExist:
        return JsonResponse({'error': 'not found'}, status=404)

    src_path  = settings.BASE_DIR / video.file
    raw_path  = DET_DIR / f'{vid_id}_raw.mp4'
    out_path  = DET_DIR / f'{vid_id}_det.mp4'

    # ― Step1: OpenCV で raw.mp4 作成 ―
    if not raw_path.exists():
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return JsonResponse({'error': 'open failed'}, status=500)

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        vw   = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while True:
            ok, frame = cap.read()
            if not ok: break
            res = detect_model(frame, verbose=False)[0]
            vw.write(res.plot())
        cap.release(); vw.release()

    # ― Step2: ffmpeg で faststart H.264 ―
    if not out_path.exists():
        cmd = [
            'ffmpeg', '-y', '-i', str(raw_path),
            '-c:v', 'libx264', '-preset', 'veryfast',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            str(out_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            return JsonResponse({'error': 'ffmpeg failed', 'detail': str(e)}, status=500)
        finally:
            raw_path.unlink(missing_ok=True)

    return JsonResponse({'ok': True, 'video_id': vid_id})

@require_http_methods(['GET'])
def get_det_video(request):
    """
    /getDetVideo/?video_id=xx  → mp4 ストリーミング (Range 対応)
    """
    try:
        vid_id = int(request.GET.get('video_id', ''))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    file_path = DET_DIR / f'{vid_id}_det.mp4'
    if not file_path.exists():
        return JsonResponse({'error': 'not generated'}, status=404)

    return FileResponse(open(file_path, 'rb'),
                        as_attachment=False,
                        filename=file_path.name,
                        content_type='video/mp4')
# >>> no06/views.py 全文ここまで <<<
