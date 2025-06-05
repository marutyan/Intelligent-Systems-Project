from django.urls import path
from . import views

urlpatterns = [
    path('',                   views.index,             name='index'),
    path('upload/',            views.upload_video,      name='upload_video'),
    path('selectVideos/',      views.select_videos,     name='select_videos'),
    path('generateData/',      views.generate_data,     name='generate_data'),
    path('make_distribution/', views.make_distribution, name='make_distribution'),
    path('showDistribution/',  views.show_distribution, name='show_distribution'),
    path('getImage/',          views.get_image,         name='get_image'),
    path('getVideo/',          views.get_video,         name='get_video'),
    path('getThumbList/',      views.get_thumb_list,    name='get_thumb_list'),

    # --- BBox detection ---
    path('selectDetect/',      views.select_detect,     name='select_detect'),
    path('bboxDetect/',        views.bbox_detect,       name='bbox_detect'),
    path('getDetVideo/',      views.get_det_video,     name='get_det_video'),

    path('showDirectory/',     views.show_directory,    name='show_directory'),
]
