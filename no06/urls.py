# >>> no06/urls.py 全文 <<<
from django.urls import path
from . import views

urlpatterns = [
    # ── 基本 ───────────────────────────────────────
    path('',                     views.index,               name='index'),
    path('upload/',              views.upload_video,        name='upload_video'),
    path('deleteVideo/<int:vid>/', views.delete_video,      name='delete_video'),

    # ★ 追加: 動画ストリーミング & 処理済み動画
    path('getVideo/',            views.get_video,           name='get_video'),
    path('getProcessedVideo/',   views.get_processed_video, name='get_processed_video'),

    # ── データ生成 / 分布 ───────────────────────────
    path('selectVideos/',        views.select_videos,       name='select_videos'),
    path('generateData/',        views.generate_data,       name='generate_data'),
    path('make_distribution/',   views.make_distribution,   name='make_distribution'),
    path('showDistribution/',    views.show_distribution,   name='show_distribution'),

    # ── ファイル・画像 ──────────────────────────────
    path('showDirectory/',       views.show_directory,      name='show_directory'),
    path('getImage/',            views.get_image,           name='get_image'),
    path('getThumbList/',        views.get_thumb_list,      name='get_thumb_list'),

    # ── BBox 検出 ───────────────────────────────────
    path('selectDetect/',        views.select_detect,       name='select_detect'),
    path('bboxDetect/',          views.bbox_detect,         name='bbox_detect'),
    path('getDetVideo/',         views.get_det_video,       name='get_det_video'),
]
# >>> no06/urls.py ここまで <<<
