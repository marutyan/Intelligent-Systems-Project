import os, json, cv2
from pathlib import Path
from datetime import datetime

from django.conf  import settings
from django.shortcuts import render
from django.http  import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .models import Video, DistributionData

from ultralytics import YOLO
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster  import KMeans
import plotly.graph_objects as go
import plotly.colors        as pc

pose_model = YOLO('yolov8n-pose.pt')

APP_DIR = settings.BASE_DIR / settings.APP_NAME
VID_DIR = APP_DIR / 'media' / 'videos'
ORI_DIR = APP_DIR / 'media' / 'originals'
POSE_DIR= APP_DIR / 'media' / 'pose'          # ★ 新ディレクトリ
for d in (VID_DIR, ORI_DIR, POSE_DIR): d.mkdir(parents=True, exist_ok=True)


# =========================================================
def index(request):
    return render(request, 'no06/index.html')

# ---------------------------------------------------------
@require_http_methods(['GET'])
def select_videos(request):
    videos = Video.objects.order_by('-uploaded')
    return render(request, 'no06/select_videos.html', {'videos': videos})

# ---------------------------------------------------------
@csrf_exempt
@require_http_methods(['GET', 'POST'])
def upload_video(request):
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
        file     = str(save_path.relative_to(settings.BASE_DIR)),   # 例 no06/media/videos/xxx.mp4
        uploaded = datetime.now(),
    )
    return JsonResponse({'message': f'"{vfile.name}" をアップロードしました。'})

# ---------------------------------------------------------
def _get_average_color(img, cx, cy, r=4):
    h, w, _ = img.shape
    x1, x2 = max(0, cx - r), min(w, cx + r)
    y1, y2 = max(0, cy - r), min(h, cy + r)
    patch  = img[y1:y2, x1:x2]
    if patch.size == 0:
        return [0, 0, 0]
    b, g, r = cv2.mean(patch)[:3]
    return [int(r), int(g), int(b)]

# ---------------------------------------------------------
@require_http_methods(['GET'])
def generate_data(request):
    ids = request.GET.get('video_ids')
    if ids:
        try: sel = [int(i) for i in ids.split(',') if i.strip()]
        except ValueError: return JsonResponse({'error':'invalid ids'}, status=400)
        videos = Video.objects.filter(id__in=sel)
    else:
        videos = Video.objects.all()

    inserted = 0
    for video in videos:
        v_path = settings.BASE_DIR / video.file
        if not v_path.exists(): continue

        cap  = cv2.VideoCapture(str(v_path))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        step = int(fps)
        fn   = 0

        while True:
            ok = cap.grab()
            if not ok: break
            if fn % step: fn += 1; continue

            _, frame = cap.retrieve()
            ts_ms = int((fn / fps) * 1000)

            res = pose_model(frame, imgsz=640, verbose=False)[0]
            kpts = getattr(res, 'keypoints', None)
            if kpts is None or kpts.xy is None: fn += 1; continue

            xy   = kpts.xy.cpu().numpy()
            conf = kpts.conf.cpu().numpy() if kpts.conf is not None else None
            if xy.shape[0]==0 or xy.shape[1]<13: fn+=1; continue

            all_pts = []
            for p in range(xy.shape[0]):             # 各人物
                pts=[]
                for idx in (5,6,11,12):              # 肩・腰
                    x,y = map(int, xy[p,idx])
                    sc  = conf[p,idx] if conf is not None else 1.0
                    if sc < .3: pts=[]; break
                    pts.append((x,y))
                if len(pts)!=4: continue

                rgb12=[]
                for cx,cy in pts: rgb12.extend(_get_average_color(frame,cx,cy))

                DistributionData.objects.update_or_create(
                    video=video, timestamp=ts_ms,
                    defaults=dict(
                        tsne_x=0, tsne_y=0, tsne_z=0,
                        feature_vec=json.dumps(rgb12),
                        cluster=-1, thumb_path=''
                    )
                )
                all_pts.extend(pts)
                inserted += 1

            if all_pts:                               # サムネイル保存
                for cx,cy in all_pts:
                    cv2.circle(frame,(cx,cy),6,(0,255,0),-1)
                th_name=f"{video.id}_{ts_ms}.jpg"
                th_path=ORI_DIR / th_name
                cv2.imwrite(str(th_path), frame)
                DistributionData.objects.filter(video=video,timestamp=ts_ms)\
                    .update(thumb_path=str(th_path.relative_to(settings.BASE_DIR)))
            fn += 1
        cap.release()

    return JsonResponse({'message': f'データ生成完了 ({inserted} 点)'})

# ---------------------------------------------------------
@require_http_methods(['GET'])
def make_distribution(request):
    ids = request.GET.get('video_ids')
    if ids:
        try: sel=[int(i) for i in ids.split(',') if i.strip()]
        except ValueError: return JsonResponse({'error':'invalid ids'}, status=400)
        qs = DistributionData.objects.filter(video_id__in=sel).order_by('id')
    else:
        qs = DistributionData.objects.order_by('id')

    if qs.count() < 5:
        return JsonResponse({'error':'データが足りません'}, status=400)

    X = np.array([json.loads(d.feature_vec) for d in qs])
    ts = TSNE(n_components=3, random_state=0, perplexity=max(5,min(30,len(X)//2)))
    X3 = ts.fit_transform(X)
    cls= KMeans(n_clusters=5, random_state=0).fit_predict(X3)

    for d,(x,y,z),c in zip(qs,X3,cls):
        d.tsne_x=float(x); d.tsne_y=float(y); d.tsne_z=float(z); d.cluster=int(c)
        d.save(update_fields=['tsne_x','tsne_y','tsne_z','cluster'])

    return JsonResponse({'message':'クラスタリング完了'})

# ---------------------------------------------------------
@require_http_methods(['GET'])
def show_distribution(request):
    ids = request.GET.get('video_ids')
    if ids:
        try: sel=[int(i) for i in ids.split(',') if i.strip()]
        except ValueError: return JsonResponse({'error':'invalid ids'}, status=400)
        qs = DistributionData.objects.select_related('video')\
                                     .filter(video_id__in=sel)
    else:
        qs = DistributionData.objects.select_related('video')

    if not qs.exists(): return JsonResponse({'error':'no data'}, status=400)

    palette = pc.qualitative.Set2
    traces  = []
    for c in range(5):
        pts = [d for d in qs if d.cluster==c]
        if not pts: continue
        traces.append(go.Scatter3d(
            mode='markers', name=f'cluster{c}',
            x=[d.tsne_x for d in pts], y=[d.tsne_y for d in pts], z=[d.tsne_z for d in pts],
            marker=dict(size=3,color=palette[c]),
            customdata=[(d.video.name,d.timestamp,d.thumb_path) for d in pts],
            hovertemplate='%{customdata[0]}-%{customdata[1]}<extra></extra>',
        ))
    fig = go.Figure(
        data   = traces,
        layout = dict(scene = dict(xaxis_title='t-SNE-X',yaxis_title='t-SNE-Y',zaxis_title='t-SNE-Z'),
                      margin=dict(l=0,r=0,b=0,t=0), hovermode='closest'))
    used = {d.video.id:d.video.name for d in qs}
    result = json.loads(fig.to_json())
    result['videos']=[{'id':vid,'name':name} for vid,name in used.items()]
    return JsonResponse(result, safe=False)

# ---------------------------------------------------------
def get_image(request):
    rel = request.GET.get('path')
    if not rel: return JsonResponse({'error':'Required param "path"'}, status=400)
    absp = settings.BASE_DIR / rel
    if not absp.exists(): return JsonResponse({'error':'file not found'}, status=404)
    return FileResponse(open(absp,'rb'), content_type='image/jpeg')

# ---------------------------------------------------------
def get_video(request):
    rel = request.GET.get('path')
    if not rel: return JsonResponse({'error':'Required param "path"'}, status=400)
    absp = settings.BASE_DIR / rel
    if not absp.exists(): return JsonResponse({'error':'file not found'}, status=404)
    return FileResponse(open(absp,'rb'), content_type='video/mp4')

# ---------------------------------------------------------
def get_processed_video(request):
    """
    GET /getProcessedVideo/?video_id=<id>
    - 既存 mp4 がある → そのまま返す
    - 無ければ originals/ から 1 fps 連結して生成
    """
    try:
        vid_id = int(request.GET.get('video_id', ''))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    out_path = PRO_DIR / f'{vid_id}_proc.mp4'
    if not out_path.exists():
        frames = sorted(ORI_DIR.glob(f'{vid_id}_*.jpg'),
                        key=lambda p: int(p.stem.split('_')[1]))
        if not frames:
            return JsonResponse({'error': 'frames not found'}, status=404)

        # 動画エンコード
        sample = cv2.imread(str(frames[0]))
        h, w, _ = sample.shape
        vw = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            1.0,                                   # 1 fps
            (w, h)
        )
        for f in frames:
            img = cv2.imread(str(f))
            vw.write(img)
        vw.release()

    # FileResponse は Range リクエスト対応（Django 3.1+）
    return FileResponse(
        open(out_path, 'rb'),
        as_attachment=False,
        filename=out_path.name,
        content_type='video/mp4'
    )

# ---------------------------------------------------------
def show_directory(request):
    app_path = APP_DIR
    ext_ok = {'.py', '.html', '.js', '.css'}
    files = []
    for root, dirs, fns in os.walk(app_path):
        dirs[:] = [d for d in dirs if not d.startswith('__')]
        for f in fns:
            if Path(f).suffix in ext_ok:
                files.append(str(Path(root).relative_to(app_path) / f))
    files.sort()
    lis = ''.join(f'<li>{Path(p)}</li>' for p in files)
    return HttpResponse(f'<h2>no06 内のファイル ({len(files)})</h2>' f'<ul style="line-height:1.4em">{lis}</ul>')

# ---------------------------------------------------------
@require_http_methods(['GET'])
def get_thumb_list(request):
    """
    /getThumbList/?video_id=<id>
    → {thumbs:[<rel_path>, …]} を JSON で返す（時刻順）
    """
    try:
        vid_id = int(request.GET.get('video_id', ''))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'invalid id'}, status=400)

    frames = sorted(
        ORI_DIR.glob(f'{vid_id}_*.jpg'),
        key=lambda p: int(p.stem.split('_')[1])          # 例 13_1999.jpg → 1999
    )
    if not frames:
        return JsonResponse({'error': 'frames not found'}, status=404)

    rel_paths = [str(p.relative_to(settings.BASE_DIR)) for p in frames]
    return JsonResponse({'thumbs': rel_paths})

# ---------------------------------------------------------
@require_http_methods(['GET'])
def select_pose(request):                       # ★ 新ビュー (選択画面)
    vids = Video.objects.order_by('-uploaded')
    return render(request, 'no06/select_pose.html', {'videos': vids})

# ---------------------------------------------------------
@require_http_methods(['GET'])
def pose_estimate(request):                     # ★ 新ビュー (処理本体)
    try:
        vid_id = int(request.GET.get('video_id',''))
    except (TypeError, ValueError):
        return JsonResponse({'error':'invalid id'}, status=400)

    try:
        video = Video.objects.get(id=vid_id)
    except Video.DoesNotExist:
        return JsonResponse({'error':'not found'}, status=404)

    src_path = settings.BASE_DIR / video.file
    out_path = POSE_DIR / f'{vid_id}_pose.mp4'

    if not out_path.exists():                   # 既にあれば再利用
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return JsonResponse({'error':'cannot open video'}, status=500)

        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        vw=cv2.VideoWriter(str(out_path), fourcc, fps, (w,h))

        while True:
            ok, frame = cap.read()
            if not ok: break
            res = pose_model(frame, verbose=False)[0]
            vw.write(res.plot())                # Skeleton 描画フレーム
        cap.release(); vw.release()

    video_url = settings.MEDIA_URL + f'pose/{out_path.name}'
    return JsonResponse({'video_url': video_url})